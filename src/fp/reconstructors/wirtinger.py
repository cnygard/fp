"""Fixed-pupil Wirtinger descent on half mean-squared intensity error."""

from math import prod

import torch

from ..config import SolverOptions
from ..data import MeasurementSet
from ..diagnostics import StateCallback, emit_state
from ..fft import fft2c
from ..optics import FourierPtychographyModel
from ..results import EpochMetrics, ReconstructionResult
from ..validation import require_positive_integer
from ._common import initialize_pupil, initialize_spectrum, make_result
from .protocol import ProgressCallback


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
            derivative[patch] += (
                2 / (intensities.numel() * adjoint_scale) * fft2c(residual * field) * pupil.conj()
            )
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

    def __init__(self, model: FourierPtychographyModel, options: SolverOptions = SolverOptions()):
        self.model = model
        self.options = options

    @torch.no_grad()
    def reconstruct(
        self,
        measurements: MeasurementSet,
        *,
        initial_field=None,
        initial_pupil=None,
        callback: ProgressCallback | None = None,
        observer: StateCallback | None = None,
        observer_interval: int = 1,
    ) -> ReconstructionResult:
        require_positive_integer("observer_interval", observer_interval)
        model, options = self.model, self.options
        model.validate_measurements(measurements)
        intensities = measurements.intensities.to(device=model.device, dtype=model.real_dtype)
        spectrum = initialize_spectrum(model, intensities, initial_field)
        pupil = initialize_pupil(model, initial_pupil)
        initial_step = self._initial_step(intensities, pupil)
        history = [EpochMetrics(0, model.amplitude_rmse(spectrum, intensities, pupil=pupil))]
        emit_state(observer, epoch=0, step=0, spectrum=spectrum, pupil=pupil, metrics=history[0])
        for epoch in range(1, options.epochs + 1):
            loss, gradient = intensity_loss_gradient(model, spectrum, intensities, pupil)
            norm_squared = gradient.abs().square().sum().item()
            if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
                raise RuntimeError("Wirtinger flow encountered nonfinite loss or gradient")
            candidate, candidate_loss, step_size = self._backtrack(
                spectrum,
                intensities,
                pupil,
                loss,
                gradient,
                norm_squared,
                initial_step,
            )
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
                    observer,
                    epoch=epoch,
                    step=epoch,
                    spectrum=spectrum,
                    pupil=pupil,
                    frame_index=0,
                    metrics=metrics,
                    extras={
                        "measurement": intensities[0],
                        "prediction": prediction,
                        "residual": intensities[0] - prediction,
                        "gradient": gradient,
                        "update": update,
                        "intensity_loss": candidate_loss,
                        "step_size": torch.tensor(
                            step_size, device=model.device, dtype=model.real_dtype
                        ),
                    },
                )
        return make_result(
            model,
            options,
            measurements,
            spectrum,
            pupil,
            history,
            algorithm="wirtinger_flow",
            pupil_recovery={"enabled": False, "support_radius_scale": 1.0},
            optimization={
                "loss": "half_mean_squared_intensity",
                "step": "curvature_scaled_backtracking",
                "initialization": "resized_mean_intensity"
                if initial_field is None
                else "supplied_field",
            },
        )

    def _initial_step(self, intensities, pupil):
        model = self.model
        # A*A is diagonal in the spectrum coordinates, including patch overlap.
        coverage = torch.zeros(model.grid.object_shape, device=model.device, dtype=model.real_dtype)
        for patch in model.patches:
            coverage[patch] += pupil.abs().square()
        operator_norm_squared = coverage.amax().item() / (
            prod(model.grid.measurement_shape) * model.grid.field_scale**2
        )
        curvature = 6 * intensities.amax().item() * operator_norm_squared / intensities.numel()
        return self.options.learning_rate / curvature if curvature > 0 else 1.0

    def _backtrack(self, spectrum, intensities, pupil, loss, gradient, norm_squared, initial_step):
        """Find a finite step satisfying sufficient intensity-loss decrease."""
        model = self.model
        step_size = initial_step
        for _ in range(80):
            candidate = spectrum - step_size * gradient
            candidate_loss, _ = intensity_loss_gradient(
                model, candidate, intensities, pupil, gradient=False
            )
            if torch.isfinite(candidate_loss) and (
                candidate_loss.item() <= loss.item() - 1e-4 * step_size * norm_squared
                or torch.equal(candidate, spectrum)
            ):
                break
            step_size *= 0.5
        else:
            raise RuntimeError("Wirtinger flow backtracking could not find a finite descent step")
        return candidate, candidate_loss, step_size
