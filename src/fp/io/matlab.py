"""MATLAB intensity cubes and LED-location calibration, with explicit units."""

from dataclasses import dataclass
from math import cos, isfinite, pi, sin
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from ..config import _positive, _positive_integer
from ..data import MeasurementSet, wavevectors_tensor
from ..geometry import led_wavevectors
from ..preprocessing import center_crop_resize


def _read_mat(path, keys):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return loadmat(path, variable_names=keys)
    except NotImplementedError as error:
        raise ValueError(f"{path}: MATLAB v7.3/HDF5 is unsupported; save as MATLAB v7 or earlier") from error


@dataclass(frozen=True)
class LEDCalibration:
    """Grid locations and optical calibration for an LED illumination array.

    MATLAB xlocation_s/ylocation_s are grid indices in acquisition order.
    A wavevectors_rad_per_m (frames,2) variable instead supplies calibrated
    wavevectors directly; no location transform or refraction is then applied.
    """

    spacing_m: float = 4e-3
    height_m: float = 90.88e-3
    glass_thickness_m: float = 1e-3
    glass_refractive_index: float = 1.52
    offset_xy_m: tuple[float, float] = (0.0, 0.0)
    rotation_deg: float = 0.0

    def __post_init__(self):
        _positive("spacing_m", self.spacing_m)
        _positive("height_m", self.height_m)
        if not isfinite(self.glass_thickness_m) or self.glass_thickness_m < 0:
            raise ValueError("glass_thickness_m must be finite and nonnegative")
        if not isfinite(self.glass_refractive_index) or self.glass_refractive_index < 1:
            raise ValueError("glass_refractive_index must be finite and at least 1")
        offset = tuple(self.offset_xy_m)
        if len(offset) != 2 or not all(isfinite(value) for value in offset):
            raise ValueError("offset_xy_m must contain finite x,y offsets")
        if not isfinite(self.rotation_deg):
            raise ValueError("rotation_deg must be finite")
        object.__setattr__(self, "offset_xy_m", offset)

    def load(self, path, wavelength_m):
        _positive("wavelength_m", wavelength_m)
        values = _read_mat(path, ["wavevectors_rad_per_m", "xlocation_s", "ylocation_s"])
        if "wavevectors_rad_per_m" in values:
            geometry = wavevectors_tensor(values["wavevectors_rad_per_m"])
            if (torch.linalg.vector_norm(geometry, dim=1) > 2 * pi / wavelength_m).any():
                raise ValueError("calibrated transverse wavevectors exceed the air wavenumber")
            return geometry
        locations = []
        for key in ("xlocation_s", "ylocation_s"):
            if key not in values:
                raise ValueError(f"{path}: missing LED calibration variable {key}")
            value = np.asarray(values[key])
            if (value.ndim > 2 or value.size == 0
                    or (value.ndim == 2 and min(value.shape) != 1)):
                raise ValueError(f"{key} must be a nonempty row or column vector")
            locations.append(value.reshape(-1))
        if locations[0].shape != locations[1].shape:
            raise ValueError("LED x/y location counts must match")
        positions = wavevectors_tensor(np.stack(locations, axis=1)) * self.spacing_m
        angle = self.rotation_deg * pi / 180
        rotation = torch.tensor([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]], dtype=torch.float64)
        positions = positions @ rotation.T + torch.tensor(self.offset_xy_m, dtype=torch.float64)
        return led_wavevectors(
            positions, self.height_m, wavelength_m,
            glass_thickness_m=self.glass_thickness_m,
            glass_refractive_index=self.glass_refractive_index,
        )


@dataclass(frozen=True)
class MatlabHDRLoader:
    """Read (height,width,frames) imlow_HDR intensities without normalization.

    intensity_scale is an explicit multiplier, including for integer MATLAB
    arrays. Values above one are valid HDR intensities. Frame order is retained.
    """

    wavevectors_rad_per_m: torch.Tensor
    pixel_size_m: float
    resolution: int = 300
    key: str = "imlow_HDR"
    intensity_scale: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "wavevectors_rad_per_m", wavevectors_tensor(self.wavevectors_rad_per_m))
        _positive("pixel_size_m", self.pixel_size_m)
        _positive_integer("resolution", self.resolution)
        _positive("intensity_scale", self.intensity_scale)
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("MATLAB intensity key must be a nonempty string")

    def load(self, path):
        path = Path(path)
        values = _read_mat(path, [self.key])
        if self.key not in values:
            raise ValueError(f"{path}: missing intensity variable {self.key}")
        images = np.asarray(values[self.key])
        if images.ndim != 3 or min(images.shape) < 1:
            raise ValueError(f"{self.key} must have shape (height, width, frames)")
        if images.shape[2] != len(self.wavevectors_rad_per_m):
            raise ValueError("MATLAB frame count must match LED calibration count")
        if (not np.issubdtype(images.dtype, np.number) or np.iscomplexobj(images)
                or not np.isfinite(images).all() or (images < 0).any()):
            raise ValueError("MATLAB intensities must be real, finite and nonnegative")
        frames = []
        for index in range(images.shape[2]):
            frame, spacing = center_crop_resize(images[:, :, index], self.resolution, self.pixel_size_m)
            frames.append(frame * self.intensity_scale)
        return MeasurementSet(
            torch.stack(frames), self.wavevectors_rad_per_m, spacing,
            tuple(f"{path.name}:{self.key}[{index}]" for index in range(images.shape[2])),
        )
