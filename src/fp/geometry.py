"""Acquisition adapters; solvers consume wavevectors rather than ambiguous angles."""

from math import isfinite, isqrt, pi

import torch

from .data import wavevectors_tensor
from .validation import require_positive, require_positive_integer


def angles_to_wavevectors(angles_rad, wavelength_m: float):
    """Angles in radians, columns (theta_x, theta_y), for array data."""
    require_positive("wavelength_m", wavelength_m)
    return wavevectors_tensor(2 * pi / wavelength_m * torch.sin(wavevectors_tensor(angles_rad)))


def spiral_order(size: int):
    """Center-right-down-left-up square traversal, including even sizes."""
    require_positive_integer("size", size)
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
    """Geometry in file order for spiral array acquisition.

    Preserve its sign and independent x/z and y/z angle convention. This is not
    an adapter for a rotating slit or a refracting LED array.
    """
    require_positive_integer("count", count)
    require_positive("translation_m", translation_m)
    require_positive("object_distance_m", object_distance_m)
    size = isqrt(count)
    if size * size != count:
        raise ValueError("array acquisition requires a perfect-square sample count")
    positions = torch.arange(size, dtype=torch.float64) - size // 2
    y, x = torch.meshgrid(positions, positions, indexing="ij")
    displacements = -torch.stack((x, y), dim=-1).reshape(-1, 2) * translation_m
    angles = torch.atan(displacements / object_distance_m)
    return angles_to_wavevectors(angles, wavelength_m)[spiral_order(size)]


def led_wavevectors(
    positions_xy_m,
    height_m: float,
    wavelength_m: float,
    *,
    glass_thickness_m: float = 0.0,
    glass_refractive_index: float = 1.52,
):
    """LED positions (x,y) -> incident (kx,ky), in the supplied frame order.

    height_m is the air gap below a plane-parallel slide. Solve Snell's law
    for a ray ending at the sample above the slide. The sample medium is air.
    Direction cosines are already sin(theta); do not apply sin a second time.
    Positive LED x/y positions produce negative incident kx/ky.
    """
    require_positive("height_m", height_m)
    require_positive("wavelength_m", wavelength_m)
    if not isfinite(glass_thickness_m) or glass_thickness_m < 0:
        raise ValueError("glass_thickness_m must be finite and nonnegative")
    if not isfinite(glass_refractive_index) or glass_refractive_index < 1:
        raise ValueError("glass_refractive_index must be finite and at least 1")
    positions = wavevectors_tensor(positions_xy_m)
    radius = torch.linalg.vector_norm(positions, dim=1)
    upper = radius / torch.hypot(radius, torch.full_like(radius, height_m))
    lower = torch.zeros_like(upper)
    # Monotone equation r = H tan(theta_air) + h tan(theta_glass).
    for _ in range(64):
        sine = (lower + upper) / 2
        in_glass = sine / glass_refractive_index
        distance = height_m * sine / torch.sqrt(
            1 - sine.square()
        ) + glass_thickness_m * in_glass / torch.sqrt(1 - in_glass.square())
        lower = torch.where(distance < radius, sine, lower)
        upper = torch.where(distance < radius, upper, sine)
    sine = (lower + upper) / 2
    direction = positions / radius.clamp_min(torch.finfo(torch.float64).tiny)[:, None]
    return wavevectors_tensor(-2 * pi / wavelength_m * direction * sine[:, None])
