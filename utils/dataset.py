"""Minimal local image discovery for the demo."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_single_ground_image(directory: Path) -> tuple[Path, Image.Image]:
    paths = _image_paths(directory)
    if not paths:
        raise FileNotFoundError(f"No supported ground image found in {directory}")
    if len(paths) > 1:
        raise ValueError(f"Use exactly one ground image in {directory}; found {len(paths)}.")
    return paths[0], _load_rgb(paths[0])


def load_aerial_gallery(directory: Path) -> list[tuple[Path, Image.Image]]:
    paths = _image_paths(directory)
    if not paths:
        raise FileNotFoundError(f"No supported aerial images found in {directory}")
    return [(path, _load_rgb(path)) for path in paths]
