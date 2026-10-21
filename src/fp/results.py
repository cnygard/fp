"""Numerical results remain complex; visualization is a separate operation."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    amplitude_rmse: float


@dataclass(frozen=True)
class ReconstructionResult:
    object_field: np.ndarray
    pupil: np.ndarray
    history: tuple[EpochMetrics, ...]
    settings: dict
    wavevectors_rad_per_m: np.ndarray
    frame_ids: tuple[str, ...]

    @property
    def amplitude(self):
        return np.abs(self.object_field)

    @property
    def phase(self):
        return np.angle(self.object_field)


def save_result(result: ReconstructionResult, path: str | Path) -> Path:
    """Save lossless arrays and JSON settings; load with np.load(allow_pickle=False)."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("result path must end in .npz")
    np.savez_compressed(
        path, object_field=result.object_field, pupil=result.pupil,
        epochs=np.asarray([entry.epoch for entry in result.history]),
        amplitude_rmse=np.asarray([entry.amplitude_rmse for entry in result.history]),
        settings_json=json.dumps(result.settings),
        wavevectors_rad_per_m=result.wavevectors_rad_per_m,
        frame_ids=np.asarray(result.frame_ids),
    )
    return path
