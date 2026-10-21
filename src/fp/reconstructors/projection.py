"""Joint object/pupil phase retrieval with optional diagnostic snapshots."""

from dataclasses import asdict

import torch

from ..config import SolverOptions, _positive_integer
from ..data import MeasurementSet
from ..diagnostics import StateCallback, emit_state
from ..fft import fft2c, ifft2c
from ..optics import FourierPtychographyModel
from ..preprocessing import resize_image
from ..results import EpochMetrics, ReconstructionResult
from .protocol import ProgressCallback


def initialize_spectrum(model, intensities, initial_field=None):
    if initial_field is None:
        # Mean-intensity initialization: SAVI, Fig. 6 and Materials and Methods.
        initial_field = resize_image(intensities.mean(dim=0), model.grid.object_shape)
    initial_field = model.as_complex(initial_field)
    if initial_field.shape != model.grid.object_shape or not torch.isfinite(initial_field).all():
        raise ValueError("initial_field must be finite and match the object grid")
    return fft2c(initial_field)


def _savi_corrections(spectrum, pupil, patch, delta, options):
    """Holloway et al., Sci. Adv. 3, e1602564 (2017), Eqs. (10)-(11).

    Both corrections use the pre-update fields. In Eq. (11) the normalization
    is max |Psi| over the ENTIRE object spectrum, not the current patch.
    Unit relaxation reproduces the published equations; tau1/tau2 are explicit.
    """
    region = spectrum[patch]
    pupil_abs, region_abs = pupil.abs(), region.abs()
    pupil_max = pupil_abs.amax().clamp_min(options.object_epsilon)
    object_max = spectrum.abs().amax().clamp_min(options.pupil_epsilon)
    object_weight = (pupil_abs / pupil_max * pupil.conj()
                     / (pupil_abs.square() + options.object_epsilon))
    pupil_weight = (region_abs / object_max * region.conj()
                    / (region_abs.square() + options.pupil_epsilon))
    return (options.learning_rate * object_weight * delta,
            options.pupil_learning_rate * pupil_weight * delta)


def _alternating_corrections(spectrum, pupil, patch, delta, options):
    """Joint alternating projection with globally normalized local corrections."""
    region = spectrum[patch]
    pupil_norm = pupil.abs().square().amax().clamp_min(options.object_epsilon)
    region_norm = region.abs().square().amax().clamp_min(options.pupil_epsilon)
    return (options.learning_rate * pupil.conj() * delta / pupil_norm,
            options.pupil_learning_rate * region.conj() * delta / region_norm)


@torch.no_grad()
def _run_projection(model, options, measurements, corrections, algorithm,
                    support_scale, initial_field, initial_pupil, callback,
                    observer, observer_interval):
    _positive_integer("observer_interval", observer_interval)
    model.validate_measurements(measurements)
    intensities = measurements.intensities.to(device=model.device, dtype=model.real_dtype)
    amplitudes = intensities.sqrt()
    spectrum = initialize_spectrum(model, intensities, initial_field)
    recovery_support = model.pupil_support_at_scale(support_scale)
    pupil = model.as_complex(model.pupil if initial_pupil is None else initial_pupil).detach().clone()
    if pupil.shape != model.grid.measurement_shape or not torch.isfinite(pupil).all():
        raise ValueError("initial_pupil must be finite and match the measurement grid")
    pupil *= recovery_support
    history = [EpochMetrics(0, model.amplitude_rmse(spectrum, intensities, pupil=pupil))]
    emit_state(observer, epoch=0, step=0, spectrum=spectrum, pupil=pupil, metrics=history[0])
    step = 0
    for epoch in range(1, options.epochs + 1):
        for index, patch in enumerate(model.patches):
            measured_field = model.field_from_spectrum(spectrum, index, pupil=pupil)
            constrained_field = torch.polar(amplitudes[index], measured_field.angle())
            delta = fft2c(constrained_field) * model.grid.field_scale - spectrum[patch] * pupil
            # Compute BOTH changes before mutating either the object or pupil.
            object_update, pupil_update = corrections(spectrum, pupil, patch, delta, options)
            spectrum[patch] += object_update
            pupil_update *= recovery_support
            pupil += pupil_update
            step += 1
            last_frame = index == len(model.patches) - 1
            metrics = None
            if last_frame:
                metrics = EpochMetrics(epoch, model.amplitude_rmse(spectrum, intensities, pupil=pupil))
                history.append(metrics)
                if callback is not None:
                    callback(metrics)
            if observer is not None and (step % observer_interval == 0 or last_frame):
                prediction = model.field_from_spectrum(spectrum, index, pupil=pupil).abs().square()
                emit_state(
                    observer, epoch=epoch, step=step, spectrum=spectrum, pupil=pupil,
                    frame_index=index, metrics=metrics,
                    extras={"measurement": intensities[index], "prediction": prediction,
                            "residual": intensities[index] - prediction,
                            "update": object_update, "pupil_update": pupil_update},
                )
    return ReconstructionResult(
        object_field=ifft2c(spectrum), pupil=pupil, history=tuple(history),
        settings={"algorithm": algorithm, "solver": asdict(options),
                  "pupil_recovery": {"enabled": True, "support_radius_scale": support_scale},
                  "optics": asdict(model.optics), "grid": asdict(model.grid),
                  "compute": {"device": str(model.device), "precision": model.compute.precision}},
        wavevectors_rad_per_m=model.wavevectors_rad_per_m.clone(), frame_ids=measurements.frame_ids,
    )


class AlternatingProjectionReconstructor:
    """Joint object/pupil alternating projection within the nominal pupil support."""

    pupil_support_scale = 1.0

    def __init__(self, model: FourierPtychographyModel, options: SolverOptions = SolverOptions()):
        self.model, self.options = model, options

    def reconstruct(self, measurements: MeasurementSet, *, initial_field=None, initial_pupil=None,
                    callback: ProgressCallback | None = None,
                    observer: StateCallback | None = None, observer_interval: int = 1):
        return _run_projection(
            self.model, self.options, measurements, _alternating_corrections,
            "alternating_projection", self.pupil_support_scale, initial_field, initial_pupil,
            callback, observer, observer_interval,
        )


class SaviReconstructor:
    """SAVI joint phase retrieval, Eqs. (6)-(11), Holloway et al. (2017).

    Support 'twice as large' (paper, p. 8) is implemented as a circular disk with
    twice the nominal radius/diameter, clipped to the measurement grid. Pupil
    initialization remains the nominal aperture. Speckle denoising is not applied.
    """

    pupil_support_scale = 2.0

    def __init__(self, model: FourierPtychographyModel, options: SolverOptions = SolverOptions()):
        self.model, self.options = model, options

    def reconstruct(self, measurements: MeasurementSet, *, initial_field=None, initial_pupil=None,
                    callback: ProgressCallback | None = None,
                    observer: StateCallback | None = None, observer_interval: int = 1):
        return _run_projection(
            self.model, self.options, measurements, _savi_corrections,
            "savi", self.pupil_support_scale, initial_field, initial_pupil,
            callback, observer, observer_interval,
        )
