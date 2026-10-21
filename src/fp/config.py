"""Immutable physical and numerical settings. All physical lengths are meters."""

from dataclasses import dataclass
from math import atan, isfinite, pi, sin


def _positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class OpticalConfig:
    """Circular aperture under the existing free-space propagation convention.

    pixel_size_m is the effective sample-plane spacing of the measurements,
    after preprocessing. Lens-plane and LED-refraction models are not implied.
    """

    pixel_size_m: float
    wavelength_m: float
    aperture_diameter_m: float
    object_distance_m: float
    refractive_index: float = 1.0003

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _positive(name, getattr(self, name))

    @property
    def wavenumber_rad_per_m(self) -> float:
        return 2 * pi / self.wavelength_m

    @property
    def numerical_aperture(self) -> float:
        return self.refractive_index * sin(
            atan(self.aperture_diameter_m / (2 * self.object_distance_m))
        )


@dataclass(frozen=True)
class ReconstructionGrid:
    measurement_shape: tuple[int, int]
    upsample: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "measurement_shape", tuple(self.measurement_shape))
        if len(self.measurement_shape) != 2:
            raise ValueError("measurement_shape must contain height and width")
        for size in self.measurement_shape:
            _positive_integer("measurement dimension", size)
        _positive_integer("upsample", self.upsample)

    @property
    def object_shape(self) -> tuple[int, int]:
        return tuple(size * self.upsample for size in self.measurement_shape)

    @property
    def field_scale(self) -> int:
        """FFT area scaling between the object and measurement grids."""
        return self.upsample**2


@dataclass(frozen=True)
class SolverOptions:
    epochs: int = 10
    learning_rate: float = 1.0
    pupil_learning_rate: float = 1.0
    object_epsilon: float = 1e-10
    pupil_epsilon: float = 1e-10

    def __post_init__(self) -> None:
        _positive_integer("epochs", self.epochs)
        _positive("learning_rate", self.learning_rate)
        _positive("pupil_learning_rate", self.pupil_learning_rate)
        _positive("object_epsilon", self.object_epsilon)
        _positive("pupil_epsilon", self.pupil_epsilon)
