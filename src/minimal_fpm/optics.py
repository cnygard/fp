"""Shared circular-pupil Fourier patch model for simulation and reconstruction."""

import numpy as np

from .config import OpticalConfig, ReconstructionGrid
from .data import MeasurementSet, wavevectors_array
from .fft import fft2c, ifft2c


def _round_away_from_zero(value: float) -> int:
    return int(np.copysign(np.floor(abs(value) + 0.5), value))


class FourierPtychographyModel:
    """Precompute frequency-patch bounds and fixed pupil support.

    Spectra are centered, array axes are (y,x), geometry columns are (kx,ky).
    A measured field is ifft2c(pupil * object_spectrum_patch) / upsample**2.
    The object pixel spacing is optics.pixel_size_m / grid.upsample.
    Only integer frequency shifts and a single wavelength are supported here.
    """

    def __init__(
        self, optics: OpticalConfig, grid: ReconstructionGrid, wavevectors_rad_per_m
    ):
        self.optics = optics
        self.grid = grid
        self.wavevectors_rad_per_m = wavevectors_array(wavevectors_rad_per_m)
        height, width = grid.measurement_shape
        out_h, out_w = grid.object_shape
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(height, optics.pixel_size_m))
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(width, optics.pixel_size_m))
        cutoff = optics.numerical_aperture * optics.wavenumber_rad_per_m
        self.pupil = (np.hypot(ky[:, None], kx[None, :]) <= cutoff).astype(
            np.complex128
        )
        self.pupil.setflags(write=False)
        dkx = 2 * np.pi / (optics.pixel_size_m * width)
        dky = 2 * np.pi / (optics.pixel_size_m * height)
        patches = []
        for index, (shift_x, shift_y) in enumerate(self.wavevectors_rad_per_m):
            left = out_w // 2 + _round_away_from_zero(-shift_x / dkx) - width // 2
            top = out_h // 2 + _round_away_from_zero(-shift_y / dky) - height // 2
            if left < 0 or top < 0 or left + width > out_w or top + height > out_h:
                raise ValueError(
                    f"frame {index}: frequency patch exceeds the object grid; increase upsample"
                )
            patches.append((slice(top, top + height), slice(left, left + width)))
        self.patches = tuple(patches)

    def validate_measurements(self, measurements: MeasurementSet) -> None:
        if measurements.intensities.shape != (
            len(self.patches),
            *self.grid.measurement_shape,
        ):
            raise ValueError("measurement shape does not match the optical model")
        if not np.array_equal(
            measurements.wavevectors_rad_per_m, self.wavevectors_rad_per_m
        ):
            raise ValueError(
                "measurement geometry/order does not match the optical model"
            )
        if not np.isclose(
            measurements.pixel_size_m, self.optics.pixel_size_m, rtol=1e-12, atol=0
        ):
            raise ValueError(
                "measurement pixel spacing does not match the optical model"
            )

    def field_from_spectrum(self, object_spectrum, frame_index: int):
        if object_spectrum.shape != self.grid.object_shape:
            raise ValueError(
                "object spectrum shape does not match the reconstruction grid"
            )
        return (
            ifft2c(object_spectrum[self.patches[frame_index]] * self.pupil)
            / self.grid.field_scale
        )

    def predict_intensities(self, object_field):
        object_field = np.asarray(object_field)
        if (
            object_field.shape != self.grid.object_shape
            or not np.isfinite(object_field).all()
        ):
            raise ValueError(
                "object field must be finite and match the reconstruction grid"
            )
        spectrum = fft2c(object_field)
        return np.stack(
            [
                np.abs(self.field_from_spectrum(spectrum, i)) ** 2
                for i in range(len(self.patches))
            ]
        )

    def amplitude_rmse(self, object_spectrum, intensities) -> float:
        error = 0.0
        for index in range(len(self.patches)):
            residual = np.abs(
                self.field_from_spectrum(object_spectrum, index)
            ) - np.sqrt(intensities[index])
            error += float(np.sum(residual**2))
        return float(np.sqrt(error / intensities.size))
