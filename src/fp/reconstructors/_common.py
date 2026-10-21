"""Initialization and result assembly shared by reconstruction algorithms."""

from dataclasses import asdict

import torch

from ..config import SolverOptions
from ..data import MeasurementSet
from ..fft import fft2c, ifft2c
from ..optics import FourierPtychographyModel
from ..preprocessing import resize_image
from ..results import EpochMetrics, ReconstructionResult


def initialize_spectrum(
    model: FourierPtychographyModel, intensities: torch.Tensor, initial_field=None
) -> torch.Tensor:
    if initial_field is None:
        # Mean-intensity initialization: SAVI, Fig. 6 and Materials and Methods.
        initial_field = resize_image(intensities.mean(dim=0), model.grid.object_shape)
    initial_field = model.as_complex(initial_field)
    if initial_field.shape != model.grid.object_shape or not torch.isfinite(initial_field).all():
        raise ValueError("initial_field must be finite and match the object grid")
    return fft2c(initial_field)


def initialize_pupil(
    model: FourierPtychographyModel, initial_pupil=None, *, support_scale: float = 1.0
) -> torch.Tensor:
    """Copy the starting pupil so solver updates cannot mutate caller state."""
    source = model.pupil if initial_pupil is None else initial_pupil
    pupil = model.as_complex(source).detach().clone()
    if pupil.shape != model.grid.measurement_shape or not torch.isfinite(pupil).all():
        raise ValueError("initial_pupil must be finite and match the measurement grid")
    pupil *= model.pupil_support_at_scale(support_scale)
    return pupil


def make_result(
    model: FourierPtychographyModel,
    options: SolverOptions,
    measurements: MeasurementSet,
    spectrum: torch.Tensor,
    pupil: torch.Tensor,
    history: list[EpochMetrics],
    *,
    algorithm: str,
    pupil_recovery: dict,
    optimization: dict | None = None,
) -> ReconstructionResult:
    settings = {
        "algorithm": algorithm,
        "solver": asdict(options),
        "pupil_recovery": pupil_recovery,
        "optics": asdict(model.optics),
        "grid": asdict(model.grid),
        "compute": {"device": str(model.device), "precision": model.compute.precision},
    }
    if optimization is not None:
        settings["optimization"] = optimization
    return ReconstructionResult(
        object_field=ifft2c(spectrum),
        pupil=pupil,
        history=tuple(history),
        settings=settings,
        wavevectors_rad_per_m=model.wavevectors_rad_per_m.clone(),
        frame_ids=measurements.frame_ids,
    )
