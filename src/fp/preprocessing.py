"""Torch image preprocessing with an explicit change in physical sampling."""

import torch.nn.functional as functional

from .config import _positive, _positive_integer
from .tensors import real_tensor


def resize_image(image, shape):
    """Bilinear half-pixel interpolation, matching legacy OpenCV resize semantics."""
    image = real_tensor(image)
    if image.ndim != 2 or min(image.shape) < 1:
        raise ValueError("expected a nonempty grayscale image")
    return functional.interpolate(
        image[None, None], size=tuple(shape), mode="bilinear", align_corners=False,
    )[0, 0]


def center_crop_resize(image, size: int, pixel_size_m: float):
    """Resize a centered square while preserving its physical field of view."""
    _positive_integer("size", size)
    _positive("pixel_size_m", pixel_size_m)
    image = real_tensor(image)
    if image.ndim != 2 or min(image.shape) < 1:
        raise ValueError("expected a nonempty grayscale image")
    height, width = image.shape
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    cropped = image[top:top + side, left:left + side]
    return resize_image(cropped, (size, size)), pixel_size_m * side / size
