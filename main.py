"""Run the compact ConGeo FoV-variation retrieval demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model.model import SimpleImageEncoder, cosine_similarity
from utils.dataset import load_aerial_gallery, load_single_ground_image
from utils.transforms import apply_fov_transform


FOVS = (360, 180, 90, 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare aerial retrieval under ground-image FoV variations."
    )
    parser.add_argument("--ground-dir", type=Path, default=Path("sample/ground"))
    parser.add_argument("--aerial-dir", type=Path, default=Path("sample/aerial"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--orientation-deg",
        type=float,
        default=0.0,
        help="Horizontal panorama rotation before cropping (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1.")

    try:
        ground_path, ground_image = load_single_ground_image(args.ground_dir)
        aerial_gallery = load_aerial_gallery(args.aerial_dir)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Input error: {error}") from error

    encoder = SimpleImageEncoder()
    aerial_embeddings = np.stack([encoder.encode(image) for _, image in aerial_gallery])
    k = min(args.top_k, len(aerial_gallery))

    print("ConGeo FoV Robustness Demo (course demo)")
    print(f"Ground query: {ground_path.name}")
    print(f"Aerial gallery: {len(aerial_gallery)} image(s)")
    print(f"Orientation shift: {args.orientation_deg:g} degrees")

    for fov in FOVS:
        query_image, crop_width = apply_fov_transform(
            ground_image, fov=fov, orientation_deg=args.orientation_deg
        )
        query_embedding = encoder.encode(query_image)
        scores = cosine_similarity(query_embedding, aerial_embeddings)
        ranking = np.argsort(scores)[::-1][:k]

        print(f"\nFoV {fov} degrees (crop width: {crop_width}px)")
        for rank, index in enumerate(ranking, start=1):
            filename = aerial_gallery[index][0].name
            print(f"  Top-{rank}: {filename}  similarity={scores[index]:.4f}")


if __name__ == "__main__":
    main()
