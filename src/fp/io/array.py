"""Load the existing numbered array-simulation files without CLI dependencies."""

from dataclasses import dataclass
from math import isclose
from pathlib import Path

import cv2
import numpy as np
import torch

from ..config import _positive, _positive_integer
from ..data import MeasurementSet, wavevectors_tensor
from ..preprocessing import center_crop_resize


@dataclass(frozen=True)
class ArraySimLoader:
    """Numbered 0000 files, with geometry supplied explicitly in file order.

    NPY intensities retain their scale. Raster integer values are divided by
    their dtype's maximum (255 for uint8, 65535 for uint16). No per-frame
    normalization or amplitude conversion is performed.
    """

    wavevectors_rad_per_m: torch.Tensor
    pixel_size_m: float
    resolution: int = 300
    file_extension: str = ".npy"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "wavevectors_rad_per_m",
            wavevectors_tensor(self.wavevectors_rad_per_m),
        )
        _positive("pixel_size_m", self.pixel_size_m)
        _positive_integer("resolution", self.resolution)
        if not self.file_extension.startswith(".") or "/" in self.file_extension:
            raise ValueError("file_extension must be a suffix such as .npy or .tif")

    def load(self, directory: str | Path) -> MeasurementSet:
        directory = Path(directory)
        images, frame_ids = [], []
        effective_spacing = None
        for index in range(len(self.wavevectors_rad_per_m)):
            path = directory / f"{index:04d}{self.file_extension}"
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.suffix.lower() == ".npy":
                image = np.load(path, allow_pickle=False)
            else:
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(f"could not decode {path}")
                if image.ndim != 2:
                    raise ValueError(
                        f"{path}: select a grayscale channel before loading"
                    )
                if np.issubdtype(image.dtype, np.integer):
                    image = image.astype(np.float64) / np.iinfo(image.dtype).max
            image, spacing = center_crop_resize(
                image, self.resolution, self.pixel_size_m
            )
            if effective_spacing is not None and not isclose(
                spacing,
                effective_spacing,
                rel_tol=1e-12,
                abs_tol=0,
            ):
                raise ValueError(
                    "frames have different physical fields of view after cropping"
                )
            effective_spacing = spacing
            images.append(image)
            frame_ids.append(path.name)
        return MeasurementSet(
            torch.stack(images),
            self.wavevectors_rad_per_m,
            effective_spacing,
            tuple(frame_ids),
        )
