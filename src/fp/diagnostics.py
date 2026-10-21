"""Algorithm-neutral detached snapshots; plotting never enters numerical kernels."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from .results import EpochMetrics


@dataclass(frozen=True)
class IterationState:
    """Owned, graph-free state at an iteration boundary, on the compute device.

    step counts algorithm updates; frame_index is optional for batch algorithms.
    extras can carry residuals, gradients, noise estimates, or pupil corrections.
    A callback may keep or modify a snapshot without affecting reconstruction.
    """

    epoch: int
    step: int
    spectrum: torch.Tensor
    pupil: torch.Tensor
    frame_index: int | None = None
    metrics: EpochMetrics | None = None
    extras: dict[str, torch.Tensor] = field(default_factory=dict)


StateCallback = Callable[[IterationState], None]


def emit_state(observer, *, epoch, step, spectrum, pupil, frame_index=None,
               metrics=None, extras=None):
    if observer is not None:
        observer(IterationState(
            epoch=epoch, step=step, spectrum=spectrum.detach().clone(),
            pupil=pupil.detach().clone(), frame_index=frame_index, metrics=metrics,
            extras={name: value.detach().clone() for name, value in (extras or {}).items()},
        ))
