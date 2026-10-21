from collections.abc import Callable
from typing import Protocol

from ..data import MeasurementSet
from ..results import EpochMetrics, ReconstructionResult

ProgressCallback = Callable[[EpochMetrics], None]


class Reconstructor(Protocol):
    def reconstruct(self, measurements: MeasurementSet, *, initial_field=None,
                    callback: ProgressCallback | None = None) -> ReconstructionResult: ...
