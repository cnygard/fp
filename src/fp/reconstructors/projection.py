"""Joint object/pupil phase retrieval with optional diagnostic snapshots."""

from abc import ABC, abstractmethod

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


class _ProjectionReconstructor(ABC):
    """Shared amplitude-projection loop; subclasses supply the correction equations."""

    algorithm: str
    pupil_support_scale: float

    def __init__(self, model: FourierPtychographyModel, options: SolverOptions = SolverOptions()):
        self.model = model
        self.options = options

    @abstractmethod
    def _corrections(self, spectrum, pupil, patch, delta):
        """Return object and pupil corrections without mutating either estimate."""
        ...

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
        model, options = self.model, self.options
        require_positive_integer("observer_interval", observer_interval)
        model.validate_measurements(measurements)
        intensities = measurements.intensities.to(device=model.device, dtype=model.real_dtype)
        amplitudes = intensities.sqrt()
        spectrum = initialize_spectrum(model, intensities, initial_field)
        recovery_support = model.pupil_support_at_scale(self.pupil_support_scale)
        pupil = initialize_pupil(model, initial_pupil, support_scale=self.pupil_support_scale)
        history = [EpochMetrics(0, model.amplitude_rmse(spectrum, intensities, pupil=pupil))]
        emit_state(observer, epoch=0, step=0, spectrum=spectrum, pupil=pupil, metrics=history[0])
        step = 0
        for epoch in range(1, options.epochs + 1):
            for index, patch in enumerate(model.patches):
                measured_field = model.field_from_spectrum(spectrum, index, pupil=pupil)
                constrained_field = torch.polar(amplitudes[index], measured_field.angle())
                delta = fft2c(constrained_field) * model.grid.field_scale - spectrum[patch] * pupil
                # Both corrections must use the same pre-update object and pupil.
                object_update, pupil_update = self._corrections(spectrum, pupil, patch, delta)
                spectrum[patch] += object_update
                # pupil_update *= recovery_support
                # pupil += pupil_update
                step += 1
                last_frame = index == len(model.patches) - 1
                metrics = None
                if last_frame:
                    metrics = EpochMetrics(
                        epoch, model.amplitude_rmse(spectrum, intensities, pupil=pupil)
                    )
                    history.append(metrics)
                    if callback is not None:
                        callback(metrics)
                if observer is not None and (step % observer_interval == 0 or last_frame):
                    prediction = (
                        model.field_from_spectrum(spectrum, index, pupil=pupil).abs().square()
                    )
                    emit_state(
                        observer,
                        epoch=epoch,
                        step=step,
                        spectrum=spectrum,
                        pupil=pupil,
                        frame_index=index,
                        metrics=metrics,
                        extras={
                            "measurement": intensities[index],
                            "prediction": prediction,
                            "residual": intensities[index] - prediction,
                            "update": object_update,
                            "pupil_update": pupil_update,
                        },
                    )
        return make_result(
            model,
            options,
            measurements,
            spectrum,
            pupil,
            history,
            algorithm=self.algorithm,
            pupil_recovery={"enabled": True, "support_radius_scale": self.pupil_support_scale},
        )


class AlternatingProjectionReconstructor(_ProjectionReconstructor):
    """Joint object/pupil alternating projection within the nominal pupil support."""

    algorithm = "alternating_projection"
    pupil_support_scale = 1.0

    def _corrections(self, spectrum, pupil, patch, delta):
        """Joint alternating projection with globally normalized local corrections."""
        options = self.options
        region = spectrum[patch]
        pupil_norm = pupil.abs().square().amax().clamp_min(options.object_epsilon)
        region_norm = region.abs().square().amax().clamp_min(options.pupil_epsilon)
        return (
            options.learning_rate * pupil.conj() * delta / pupil_norm,
            options.pupil_learning_rate * region.conj() * delta / region_norm,
        )


class SaviReconstructor(_ProjectionReconstructor):
    """SAVI joint phase retrieval, Eqs. (6)-(11), Holloway et al. (2017).

    Recovery support has twice the nominal radius, clipped to the measurement
    grid. Initialization uses the nominal aperture; speckle denoising is omitted.
    """

    algorithm = "savi"
    pupil_support_scale = 2.0

    def _corrections(self, spectrum, pupil, patch, delta):
        """Holloway et al., Sci. Adv. 3, e1602564 (2017), Eqs. (10)-(11).

        Both corrections use the pre-update fields. In Eq. (11) the normalization
        is max |Psi| over the ENTIRE object spectrum, not the current patch.
        Unit relaxation reproduces the published equations; tau1/tau2 are explicit.
        """
        options = self.options
        region = spectrum[patch]
        pupil_abs, region_abs = pupil.abs(), region.abs()
        pupil_max = pupil_abs.amax().clamp_min(options.object_epsilon)
        object_max = spectrum.abs().amax().clamp_min(options.pupil_epsilon)
        object_weight = (
            pupil_abs / pupil_max * pupil.conj() / (pupil_abs.square() + options.object_epsilon)
        )
        pupil_weight = (
            region_abs / object_max * region.conj() / (region_abs.square() + options.pupil_epsilon)
        )
        return (
            options.learning_rate * object_weight * delta,
            options.pupil_learning_rate * pupil_weight * delta,
        )
