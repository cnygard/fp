"""Small validation helpers shared by physical and numerical settings."""

from math import isfinite


def require_positive(name: str, value) -> None:
    if not isinstance(value, (float, int)) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
