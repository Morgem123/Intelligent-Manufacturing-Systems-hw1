"""Panorama orientation shift and horizontal FoV crop."""

from __future__ import annotations

import numpy as np
from PIL import Image


def apply_fov_transform(
    image: Image.Image, fov: int, orientation_deg: float = 0.0
) -> tuple[Image.Image, int]:
    """Shift a panorama horizontally, crop its requested FoV, then resize.

    The operation assumes the image is an equirectangular-like panorama where
    the horizontal axis spans 360 degrees. It returns a result at the input
    dimensions and the crop width before resizing.
    """
    if fov not in (70, 90, 180, 360):
        raise ValueError("FoV must be one of 70, 90, 180, or 360 degrees.")
    image = image.convert("RGB")
    width, height = image.size
    if width < 2:
        raise ValueError("Ground panorama must be at least 2 pixels wide.")

    shift_px = round((orientation_deg % 360) / 360 * width)
    array = np.asarray(image)
    shifted = np.roll(array, -shift_px, axis=1)
    crop_width = width if fov == 360 else max(1, round(width * fov / 360))
    crop = Image.fromarray(shifted[:, :crop_width])
    return crop.resize((width, height), Image.Resampling.BILINEAR), crop_width
