"""Image preprocessing with an explicit change in physical sampling."""

import cv2
import numpy as np

from .config import _positive, _positive_integer


def center_crop_resize(image, size: int, pixel_size_m: float):
    """Return a square float image and its new sample-plane pixel spacing.

    The centered square crop preserves pixel spacing; resizing preserves that
    crop's field of view and changes the spacing by crop_width / output_width.
    """
    _positive_integer("size", size)
    _positive("pixel_size_m", pixel_size_m)
    if np.iscomplexobj(image):
        raise ValueError("measurement image must be real")
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 1:
        raise ValueError("expected a nonempty grayscale image")
    height, width = image.shape
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    cropped = image[top:top + side, left:left + side]
    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized, pixel_size_m * side / size
