"""Fourier ptychography with explicit data, optics, and reconstruction objects."""

from .config import OpticalConfig, ReconstructionGrid, SolverOptions
from .data import MeasurementSet
from .optics import FourierPtychographyModel
from .reconstructors import AlternatingProjectionReconstructor, SaviReconstructor
from .results import ReconstructionResult

__all__ = [
    "OpticalConfig", "ReconstructionGrid", "SolverOptions", "MeasurementSet",
    "FourierPtychographyModel", "AlternatingProjectionReconstructor",
    "SaviReconstructor", "ReconstructionResult",
]


def main() -> None:
    """Run the CLI without parsing arguments when the package is imported."""
    from .cli import main as cli_main

    cli_main()
