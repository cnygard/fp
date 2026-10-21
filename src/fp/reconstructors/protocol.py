from collections.abc import Callable
from typing import Protocol

from ..data import MeasurementSet
from ..diagnostics import StateCallback
from ..results import EpochMetrics, ReconstructionResult

ProgressCallback = Callable[[EpochMetrics], None]


class Reconstructor(Protocol):
    """Algorithms own their iteration strategy; only inputs and reporting agree."""

    def reconstruct(
        self,
        measurements: MeasurementSet,
        *,
        initial_field=None,
        initial_pupil=None,
        callback: ProgressCallback | None = None,
        observer: StateCallback | None = None,
        observer_interval: int = 1,
    ) -> ReconstructionResult: ...
