"""Measurement tensors and geometry always travel in the same frame order."""

from dataclasses import dataclass

import torch

from .tensors import real_tensor
from .validation import require_positive


def wavevectors_tensor(values) -> torch.Tensor:
    """Owned CPU float64 geometry in (kx, ky) rad/m; rounding stays reproducible."""
    result = real_tensor(values, device="cpu", dtype=torch.float64).detach().clone()
    if result.ndim != 2 or result.shape[1] != 2 or not len(result):
        raise ValueError("wavevectors must have nonempty shape (frames, 2), ordered (kx, ky)")
    if not torch.isfinite(result).all():
        raise ValueError("wavevectors must be finite")
    return result


@dataclass(frozen=True)
class MeasurementSet:
    """Nonnegative intensity tensors (frames, height, width), never amplitudes.

    Inputs are copied and detached. Torch tensors themselves are mutable: treat
    these owned tensors as read-only, and use reordered()/to() for new datasets.
    Geometry stays on CPU in float64; image tensors may live on an accelerator.
    """

    intensities: torch.Tensor
    wavevectors_rad_per_m: torch.Tensor
    pixel_size_m: float
    frame_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        intensities = real_tensor(self.intensities).detach().clone()
        geometry = wavevectors_tensor(self.wavevectors_rad_per_m)
        if intensities.ndim != 3 or min(intensities.shape) < 1:
            raise ValueError("intensities must have nonempty shape (frames, height, width)")
        if intensities.shape[0] != len(geometry):
            raise ValueError("each intensity frame needs one wavevector")
        if not torch.isfinite(intensities).all() or (intensities < 0).any():
            raise ValueError("intensities must be finite and nonnegative")
        require_positive("pixel_size_m", self.pixel_size_m)
        ids = tuple(self.frame_ids) or tuple(str(i) for i in range(len(geometry)))
        if len(ids) != len(geometry) or len(set(ids)) != len(ids):
            raise ValueError("frame_ids must be unique and match the number of frames")
        object.__setattr__(self, "intensities", intensities)
        object.__setattr__(self, "wavevectors_rad_per_m", geometry)
        object.__setattr__(self, "frame_ids", ids)

    def to(self, device=None, dtype=None) -> MeasurementSet:
        return MeasurementSet(
            self.intensities.to(device=device, dtype=dtype),
            self.wavevectors_rad_per_m,
            self.pixel_size_m,
            self.frame_ids,
        )

    def reordered(self, order) -> MeasurementSet:
        """Apply one complete permutation to both frames and geometry."""
        order = torch.as_tensor(order, device="cpu")
        if (
            order.shape != (len(self.frame_ids),)
            or order.dtype not in (torch.int32, torch.int64)
            or not torch.equal(order.sort().values, torch.arange(len(order)))
        ):
            raise ValueError("order must be a permutation of every frame index")
        return MeasurementSet(
            self.intensities[order.to(self.intensities.device)],
            self.wavevectors_rad_per_m[order],
            self.pixel_size_m,
            tuple(self.frame_ids[i] for i in order.tolist()),
        )
