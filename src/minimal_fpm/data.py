"""Measurement arrays and geometry always travel in the same frame order."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import _positive


def wavevectors_array(values) -> NDArray[np.float64]:
    """Copy (kx, ky) angular spatial frequencies, in rad/m, as read-only data."""
    if np.iscomplexobj(values):
        raise ValueError("wavevectors must be real")
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim != 2 or result.shape[1] != 2 or not len(result):
        raise ValueError("wavevectors must have nonempty shape (frames, 2), ordered (kx, ky)")
    if not np.isfinite(result).all():
        raise ValueError("wavevectors must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MeasurementSet:
    """Nonnegative intensities (frames, height, width), never field amplitudes.

    pixel_size_m is the effective spacing after cropping/resizing. Each frame ID
    identifies the matching row of wavevectors_rad_per_m, whose columns are x,y.
    Arrays are owned snapshots so a solver cannot accidentally edit its inputs.
    """

    intensities: NDArray[np.float64]
    wavevectors_rad_per_m: NDArray[np.float64]
    pixel_size_m: float
    frame_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if np.iscomplexobj(self.intensities):
            raise ValueError("intensities must be real")
        intensities = np.array(self.intensities, dtype=np.float64, copy=True)
        geometry = wavevectors_array(self.wavevectors_rad_per_m)
        if intensities.ndim != 3 or min(intensities.shape) < 1:
            raise ValueError("intensities must have nonempty shape (frames, height, width)")
        if intensities.shape[0] != len(geometry):
            raise ValueError("each intensity frame needs one wavevector")
        if not np.isfinite(intensities).all() or (intensities < 0).any():
            raise ValueError("intensities must be finite and nonnegative")
        _positive("pixel_size_m", self.pixel_size_m)
        ids = tuple(self.frame_ids) or tuple(str(i) for i in range(len(geometry)))
        if len(ids) != len(geometry) or len(set(ids)) != len(ids):
            raise ValueError("frame_ids must be unique and match the number of frames")
        intensities.setflags(write=False)
        object.__setattr__(self, "intensities", intensities)
        object.__setattr__(self, "wavevectors_rad_per_m", geometry)
        object.__setattr__(self, "frame_ids", ids)

    def reordered(self, order) -> "MeasurementSet":
        """Apply one complete permutation to both frames and geometry."""
        order = np.asarray(order)
        if (order.shape != (len(self.frame_ids),)
                or not np.issubdtype(order.dtype, np.integer)
                or not np.array_equal(np.sort(order), np.arange(len(order)))):
            raise ValueError("order must be a permutation of every frame index")
        return MeasurementSet(
            self.intensities[order], self.wavevectors_rad_per_m[order],
            self.pixel_size_m, tuple(self.frame_ids[i] for i in order),
        )
