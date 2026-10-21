"""Torch projection algorithms with local state and optional diagnostic snapshots."""

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
        # Preserve mean-intensity initialization for parity with the prototype.
        initial_field = resize_image(intensities.mean(dim=0), model.grid.object_shape)
    initial_field = model.as_complex(initial_field)
    if (
        initial_field.shape != model.grid.object_shape
        or not torch.isfinite(initial_field).all()
    ):
        raise ValueError("initial_field must be finite and match the object grid")
    return fft2c(initial_field)


@torch.no_grad()
def _run_projection(
    model,
    options,
    measurements,
    weight,
    algorithm,
    initial_field,
    callback,
    observer,
    observer_interval,
):
    _positive_integer("observer_interval", observer_interval)
    model.validate_measurements(measurements)
    intensities = measurements.intensities.to(
        device=model.device, dtype=model.real_dtype
    )
    amplitudes = intensities.sqrt()
    spectrum = initialize_spectrum(model, intensities, initial_field)
    pupil = model.pupil.clone()
    history = [EpochMetrics(0, model.amplitude_rmse(spectrum, intensities, pupil=pupil))]
    emit_state(
        observer, epoch=0, step=0, spectrum=spectrum, pupil=pupil, metrics=history[0]
    )
    step = 0
    for epoch in range(1, options.epochs + 1):
        for index, patch in enumerate(model.patches):
            measured_field = model.field_from_spectrum(spectrum, index, pupil=pupil)
            constrained_field = torch.polar(amplitudes[index], measured_field.angle())
            delta = (
                fft2c(constrained_field) * model.grid.field_scale
                - spectrum[patch] * pupil
            )
            correction = weight * delta
            spectrum[patch] += correction
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
                    model.field_from_spectrum(spectrum, index, pupil=pupil)
                    .abs()
                    .square()
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
                        "update": correction,
                    },
                )
    return ReconstructionResult(
        object_field=ifft2c(spectrum),
        pupil=pupil,
        history=tuple(history),
        settings={
            "algorithm": algorithm,
            "solver": asdict(options),
            "optics": asdict(model.optics),
            "grid": asdict(model.grid),
            "compute": {
                "device": str(model.device),
                "precision": model.compute.precision,
            },
        },
        wavevectors_rad_per_m=model.wavevectors_rad_per_m.clone(),
        frame_ids=measurements.frame_ids,
    )


class AlternatingProjectionReconstructor:
    """Legacy MATLAB-style object update, without its separate momentum stage."""

    def __init__(
        self, model: FourierPtychographyModel, options: SolverOptions = SolverOptions()
    ):
        self.model, self.options = model, options

    def reconstruct(
        self,
        measurements: MeasurementSet,
        *,
        initial_field=None,
        callback: ProgressCallback | None = None,
        observer: StateCallback | None = None,
        observer_interval: int = 1,
    ):
        pupil = self.model.pupil
        weight = self.options.learning_rate * pupil.conj() / pupil.abs().square().max()
        return _run_projection(
            self.model,
            self.options,
            measurements,
            weight,
            "alternating_projection",
            initial_field,
            callback,
            observer,
            observer_interval,
        )


class SaviReconstructor:
    """Keep the prototype's two object corrections using the same residual.

    Pupil recovery and denoising remain inactive. Both current solvers keep
    their pupil fixed; observers also support future changing-pupil algorithms.
    """

    def __init__(
        self, model: FourierPtychographyModel, options: SolverOptions = SolverOptions()
    ):
        self.model, self.options = model, options

    def reconstruct(
        self,
        measurements: MeasurementSet,
        *,
        initial_field=None,
        callback: ProgressCallback | None = None,
        observer: StateCallback | None = None,
        observer_interval: int = 1,
    ):
        pupil = self.model.pupil
        magnitude = pupil.abs()
        weight = (
            magnitude / magnitude.max() * pupil.conj() / (magnitude.square() + 1e-10)
            + self.options.learning_rate * pupil.conj() / magnitude.square().max()
        )
        return _run_projection(
            self.model,
            self.options,
            measurements,
            weight,
            "savi",
            initial_field,
            callback,
            observer,
            observer_interval,
        )
