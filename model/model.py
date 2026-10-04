"""A tiny deterministic encoder used only to demonstrate retrieval plumbing.

This is intentionally not an implementation or checkpoint of ConGeo. Replace
``SimpleImageEncoder`` with a permitted pretrained CVGL encoder when available.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


class SimpleImageEncoder:
    """Encode a PIL image into a normalized coarse colour-layout descriptor."""

    def __init__(self, grid_width: int = 16, grid_height: int = 8) -> None:
        self.size = (grid_width, grid_height)

    def encode(self, image: Image.Image) -> np.ndarray:
        thumbnail = image.convert("RGB").resize(self.size, Image.Resampling.BILINEAR)
        vector = np.asarray(thumbnail, dtype=np.float32).reshape(-1) / 255.0
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Cannot encode an entirely zero image.")
        return vector / norm


def cosine_similarity(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Return cosine scores for a normalized query and normalized gallery."""
    if query.ndim != 1 or gallery.ndim != 2 or query.shape[0] != gallery.shape[1]:
        raise ValueError("Embedding shapes are incompatible for cosine similarity.")
    return gallery @ query
