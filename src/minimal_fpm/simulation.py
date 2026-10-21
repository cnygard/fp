"""Synthetic acquisitions use exactly the same model as reconstruction."""

from .data import MeasurementSet
from .optics import FourierPtychographyModel


def simulate_measurements(object_field, model: FourierPtychographyModel) -> MeasurementSet:
    return MeasurementSet(
        model.predict_intensities(object_field), model.wavevectors_rad_per_m,
        model.optics.pixel_size_m,
    )
