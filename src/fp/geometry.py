"""Acquisition adapters; solvers consume wavevectors rather than ambiguous angles."""

from math import isqrt, pi

import torch

from .config import _positive, _positive_integer
from .data import wavevectors_tensor


def angles_to_wavevectors(angles_rad, wavelength_m: float):
    """Angles in radians, columns (theta_x, theta_y), matching legacy array data."""
    _positive("wavelength_m", wavelength_m)
    return wavevectors_tensor(
        2 * pi / wavelength_m * torch.sin(wavevectors_tensor(angles_rad))
    )


def spiral_order(size: int):
    """Center-right-down-left-up square traversal, including even sizes."""
    _positive_integer("size", size)
    row = col = size // 2
    order = [row * size + col]
    step, direction = 1, 0
    directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
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
    return torch.tensor(order, dtype=torch.int64)


def array_wavevectors(
    count: int, translation_m: float, object_distance_m: float, wavelength_m: float
):
    """Geometry in file order for the legacy spiral array acquisition.

    Preserve its sign and independent x/z and y/z angle convention. This is not
    an adapter for a rotating slit or a refracting LED array.
    """
    _positive_integer("count", count)
    _positive("translation_m", translation_m)
    _positive("object_distance_m", object_distance_m)
    size = isqrt(count)
    if size * size != count:
        raise ValueError("array acquisition requires a perfect-square sample count")
    positions = torch.arange(size, dtype=torch.float64) - size // 2
    y, x = torch.meshgrid(positions, positions, indexing="ij")
    displacements = -torch.stack((x, y), dim=-1).reshape(-1, 2) * translation_m
    angles = torch.atan(displacements / object_distance_m)
    return angles_to_wavevectors(angles, wavelength_m)[spiral_order(size)]
