"""Results stay as complex Torch tensors; NumPy is only a portable export boundary."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    amplitude_rmse: float


@dataclass(frozen=True)
class ReconstructionResult:
    object_field: torch.Tensor
    pupil: torch.Tensor
    history: tuple[EpochMetrics, ...]
    settings: dict
    wavevectors_rad_per_m: torch.Tensor
    frame_ids: tuple[str, ...]

    @property
    def amplitude(self) -> torch.Tensor:
        return self.object_field.abs()

    @property
    def phase(self) -> torch.Tensor:
        return self.object_field.angle()


def to_numpy(tensor):
    """Explicit device/graph boundary for file formats and presentation."""
    return tensor.detach().cpu().resolve_conj().resolve_neg().numpy()


def save_result(result: ReconstructionResult, path: str | Path) -> Path:
    import numpy as np

    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("result path must end in .npz")
    np.savez_compressed(
        path,
        object_field=to_numpy(result.object_field),
        pupil=to_numpy(result.pupil),
        epochs=np.asarray([entry.epoch for entry in result.history]),
        amplitude_rmse=np.asarray([entry.amplitude_rmse for entry in result.history]),
        settings_json=json.dumps(result.settings),
        wavevectors_rad_per_m=to_numpy(result.wavevectors_rad_per_m),
        frame_ids=np.asarray(result.frame_ids),
    )
    return path
