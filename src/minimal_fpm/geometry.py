"""Acquisition adapters; solvers consume wavevectors rather than ambiguous angles."""

from math import isqrt

import numpy as np

from .config import _positive, _positive_integer
from .data import wavevectors_array


def angles_to_wavevectors(angles_rad, wavelength_m: float):
    """Angles in radians with columns (theta_x, theta_y), matching legacy array data."""
    _positive("wavelength_m", wavelength_m)
    angles = wavevectors_array(angles_rad)
    return wavevectors_array(2 * np.pi / wavelength_m * np.sin(angles))


def spiral_order(size: int):
    """Legacy center-right-down-left-up square traversal, including even sizes."""
    _positive_integer("size", size)
    row = col = size // 2
    order = [row * size + col]
    step = 1
    directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
    direction = 0
    while len(order) < size * size:
        for _ in range(2):
            dr, dc = directions[direction % 4]
            for _ in range(step):
                row += dr
                col += dc
                if 0 <= row < size and 0 <= col < size:
                    order.append(row * size + col)
            direction += 1
        step += 1
    return np.asarray(order, dtype=np.intp)


def array_wavevectors(count: int, translation_m: float,
                      object_distance_m: float, wavelength_m: float):
    """Geometry for files already acquired in the legacy array's spiral order.

    Preserve the legacy sign and independent x/z and y/z angle convention.
    This is not an adapter for a rotating slit or a refracting LED array.
    """
    _positive_integer("count", count)
    _positive("translation_m", translation_m)
    _positive("object_distance_m", object_distance_m)
    size = isqrt(count)
    if size * size != count:
        raise ValueError("array acquisition requires a perfect-square sample count")
    positions = np.arange(size) - size // 2
    x, y = np.meshgrid(positions, positions, indexing="xy")
    displacements = -np.stack((x, y), axis=-1).reshape(-1, 2) * translation_m
    angles = np.arctan(displacements / object_distance_m)
    return angles_to_wavevectors(angles, wavelength_m)[spiral_order(size)]
