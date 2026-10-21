"""Fixed-pupil Wirtinger descent on half mean-squared intensity error."""

from dataclasses import asdict
from math import prod

import torch

from ..config import SolverOptions, _positive_integer
from ..diagnostics import emit_state
from ..fft import fft2c, ifft2c
from ..results import EpochMetrics, ReconstructionResult
from .projection import initialize_spectrum


def intensity_loss_gradient(model, spectrum, intensities, pupil, *, gradient=True):
    """Streaming equivalent of intensity_loss and its complex autograd gradient.

    For L = || |Az|^2 - I ||^2 / (2M), Torch's complex gradient is
    2 A*((|Az|^2-I) Az) / M. FFT/upsampling factors match model.adjoint_fields.
    Only one measurement-sized field is allocated at a time.
    """
    loss = torch.zeros((), device=model.device, dtype=model.real_dtype)
    derivative = torch.zeros_like(spectrum) if gradient else None
    adjoint_scale = prod(model.grid.measurement_shape) * model.grid.field_scale
    for index, patch in enumerate(model.patches):
        field = model.field_from_spectrum(spectrum, index, pupil=pupil)
        residual = field.abs().square() - intensities[index]
        loss += residual.square().sum() / (2 * intensities.numel())
        if gradient:
            derivative[patch] += (2 / (intensities.numel() * adjoint_scale)
                                  * fft2c(residual * field) * pupil.conj())
    return loss, derivative


class WirtingerFlowReconstructor:
    """Batch intensity-loss descent with a fixed pupil and backtracking.

    Uses the package's resized mean-intensity initialization (or initial_field),
    rather than the random-measurement spectral initializer of Candes et al.
    Learning rate multiplies a curvature-scaled starting step. Backtracking
    enforces sufficient intensity-loss decrease; amplitude RMSE is diagnostic.
    One epoch is one full-data gradient update. No measurement re-squaring or
    fitted noise term is applied. initial_pupil supplies a fixed pupil estimate.
    """

    pupil_support_scale = 1.0

    def __init__(self, model, options: SolverOptions = SolverOptions()):
        self.model, self.options = model, options

    @torch.no_grad()
    def reconstruct(self, measurements, *, initial_field=None, initial_pupil=None,
                    callback=None, observer=None, observer_interval=1):
        _positive_integer("observer_interval", observer_interval)
        model, options = self.model, self.options
        model.validate_measurements(measurements)
        intensities = measurements.intensities.to(device=model.device, dtype=model.real_dtype)
        spectrum = initialize_spectrum(model, intensities, initial_field)
        pupil = model.as_complex(model.pupil if initial_pupil is None else initial_pupil).detach().clone()
        if pupil.shape != model.grid.measurement_shape or not torch.isfinite(pupil).all():
            raise ValueError("initial_pupil must be finite and match the measurement grid")
        pupil *= model.pupil_support
        # A*A is diagonal in the spectrum coordinates, including patch overlap.
        coverage = torch.zeros(model.grid.object_shape, device=model.device, dtype=model.real_dtype)
        for patch in model.patches:
            coverage[patch] += pupil.abs().square()
        operator_norm_squared = coverage.amax().item() / (
            prod(model.grid.measurement_shape) * model.grid.field_scale**2)
        curvature = 6 * intensities.amax().item() * operator_norm_squared / intensities.numel()
        initial_step = options.learning_rate / curvature if curvature > 0 else 1.0
        history = [EpochMetrics(0, model.amplitude_rmse(spectrum, intensities, pupil=pupil))]
        emit_state(observer, epoch=0, step=0, spectrum=spectrum, pupil=pupil, metrics=history[0])
        for epoch in range(1, options.epochs + 1):
            loss, gradient = intensity_loss_gradient(model, spectrum, intensities, pupil)
            norm_squared = gradient.abs().square().sum().item()
            if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
                raise RuntimeError("Wirtinger flow encountered nonfinite loss or gradient")
            step_size = initial_step
            for _ in range(80):
                candidate = spectrum - step_size * gradient
                candidate_loss, _ = intensity_loss_gradient(
                    model, candidate, intensities, pupil, gradient=False)
                if (torch.isfinite(candidate_loss)
                        and (candidate_loss.item() <= loss.item() - 1e-4 * step_size * norm_squared
                             or torch.equal(candidate, spectrum))):
                    break
                step_size *= 0.5
            else:
                raise RuntimeError("Wirtinger flow backtracking could not find a finite descent step")
            update = candidate - spectrum if observer is not None else None
            spectrum = candidate
            metrics = EpochMetrics(epoch, model.amplitude_rmse(spectrum, intensities, pupil=pupil))
            history.append(metrics)
            if callback is not None:
                callback(metrics)
            if observer is not None:
                # Every batch update is also an epoch boundary. Frame zero is
                # the representative measurement in the diagnostic image panels.
                prediction = model.field_from_spectrum(spectrum, 0, pupil=pupil).abs().square()
                emit_state(
                    observer, epoch=epoch, step=epoch, spectrum=spectrum, pupil=pupil,
                    frame_index=0, metrics=metrics,
                    extras={"measurement": intensities[0], "prediction": prediction,
                            "residual": intensities[0] - prediction, "gradient": gradient,
                            "update": update, "intensity_loss": candidate_loss,
                            "step_size": torch.tensor(step_size, device=model.device, dtype=model.real_dtype)},
                )
        return ReconstructionResult(
            object_field=ifft2c(spectrum), pupil=pupil, history=tuple(history),
            settings={"algorithm": "wirtinger_flow", "solver": asdict(options),
                      "pupil_recovery": {"enabled": False, "support_radius_scale": 1.0},
                      "optimization": {"loss": "half_mean_squared_intensity", "step": "curvature_scaled_backtracking",
                                       "initialization": "resized_mean_intensity" if initial_field is None else "supplied_field"},
                      "optics": asdict(model.optics), "grid": asdict(model.grid),
                      "compute": {"device": str(model.device), "precision": model.compute.precision}},
            wavevectors_rad_per_m=model.wavevectors_rad_per_m.clone(), frame_ids=measurements.frame_ids,
        )
