"""Projection algorithms with local iteration state and shared optical operations."""

from dataclasses import asdict

import cv2
import numpy as np

from ..config import SolverOptions
from ..data import MeasurementSet
from ..fft import fft2c, ifft2c
from ..optics import FourierPtychographyModel
from ..results import EpochMetrics, ReconstructionResult
from .protocol import ProgressCallback


def _run_projection(model, options, measurements, update_weight, algorithm,
                    initial_field, callback):
    model.validate_measurements(measurements)
    if initial_field is None:
        # Preserve the existing solvers' mean-intensity initialization for parity.
        height, width = model.grid.object_shape
        initial_field = cv2.resize(measurements.intensities.mean(axis=0), (width, height))
    initial_field = np.asarray(initial_field, dtype=np.complex128)
    if initial_field.shape != model.grid.object_shape or not np.isfinite(initial_field).all():
        raise ValueError("initial_field must be finite and match the object grid")
    spectrum = fft2c(initial_field)
    history = [EpochMetrics(0, model.amplitude_rmse(spectrum, measurements.intensities))]
    for epoch in range(1, options.epochs + 1):
        for index, patch in enumerate(model.patches):
            measured_field = model.field_from_spectrum(spectrum, index)
            constrained_field = np.sqrt(measurements.intensities[index]) * np.exp(1j * np.angle(measured_field))
            delta = (fft2c(constrained_field) * model.grid.field_scale
                     - spectrum[patch] * model.pupil)
            spectrum[patch] += update_weight * delta
        metrics = EpochMetrics(epoch, model.amplitude_rmse(spectrum, measurements.intensities))
        history.append(metrics)
        if callback is not None:
            callback(metrics)
    return ReconstructionResult(
        object_field=ifft2c(spectrum), pupil=model.pupil.copy(), history=tuple(history),
        settings={"algorithm": algorithm, "solver": asdict(options),
                  "optics": asdict(model.optics), "grid": asdict(model.grid)},
        wavevectors_rad_per_m=model.wavevectors_rad_per_m.copy(),
        frame_ids=measurements.frame_ids,
    )


class AlternatingProjectionReconstructor:
    """The legacy MATLAB-style object update, without its separate momentum stage."""

    def __init__(self, model: FourierPtychographyModel, options: SolverOptions = SolverOptions()):
        self.model = model
        self.options = options

    def reconstruct(self, measurements: MeasurementSet, *, initial_field=None,
                    callback: ProgressCallback | None = None) -> ReconstructionResult:
        pupil = self.model.pupil
        weight = self.options.learning_rate * pupil.conj() / np.max(np.abs(pupil) ** 2)
        return _run_projection(self.model, self.options, measurements, weight,
                               "alternating_projection", initial_field, callback)


class SaviReconstructor:
    """Preserve the current SAVI prototype's TWO object corrections per frame.

    Both corrections use the same pre-update residual, so their weights add.
    The commented-out pupil estimation and denoising are not active here.
    This class preserves the repository prototype, not a complete SAVI method.
    """

    def __init__(self, model: FourierPtychographyModel, options: SolverOptions = SolverOptions()):
        self.model = model
        self.options = options

    def reconstruct(self, measurements: MeasurementSet, *, initial_field=None,
                    callback: ProgressCallback | None = None) -> ReconstructionResult:
        pupil = self.model.pupil
        magnitude = np.abs(pupil)
        weight = (magnitude / magnitude.max() * pupil.conj() / (magnitude ** 2 + 1e-10)
                  + self.options.learning_rate * pupil.conj() / np.max(magnitude ** 2))
        return _run_projection(self.model, self.options, measurements, weight,
                               "savi", initial_field, callback)
