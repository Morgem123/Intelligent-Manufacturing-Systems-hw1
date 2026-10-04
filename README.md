# ConGeo FoV Robustness Demo

A compact, course-oriented demo of **field-of-view (FoV) variation** for cross-view geo-localization (CVGL). It takes one ground-view panorama, creates 360°, 180°, 90°, and 70° views, extracts a feature for each view, then ranks a small aerial-image gallery with cosine similarity.

> **Scope and attribution.** This project is based on the experimental idea and FoV settings of the official [ConGeo implementation](https://github.com/eceo-epfl/ConGeo) for *ConGeo: Robust Cross-view Geo-localization across Ground View Variations* (ECCV 2024). It is a small **course demo**, not the official repository, a reproduction of the full paper, or a from-scratch implementation of ConGeo. It deliberately does not include ConGeo training code, benchmark datasets, or official pretrained weights.

## What the demo does

```text
ground query panorama
        ↓
orientation shift + FoV crop (360 / 180 / 90 / 70)
        ↓
feature extraction
        ↓
cosine similarity against aerial gallery
        ↓
Top-K aerial retrieval for each FoV
```

The included `SimpleImageEncoder` is a small deterministic colour-and-layout descriptor so that the pipeline is transparent and lightweight. It is **not** a trained ConGeo model. Its class boundary makes it easy to replace with an authorised pretrained encoder later; do not compare its scores to paper results.

## Project layout

```text
.
├── main.py                 # single entry point
├── model/model.py          # lightweight, replaceable feature encoder
├── utils/dataset.py        # image discovery and loading
├── utils/transforms.py     # orientation shift and FoV transformation
├── sample/
│   ├── ground/             # put one panorama here
│   └── aerial/             # put candidate aerial images here
├── Dockerfile
├── requirements.txt
└── .gitignore
```

## Quick start

1. Put one `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp` ground panorama in `sample/ground/`.
2. Put two or more candidate aerial images in `sample/aerial/`.
3. Run:

```bash
python main.py
```

For a different gallery location or a fixed panorama rotation:

```bash
python main.py --ground-dir sample/ground --aerial-dir sample/aerial --top-k 3 --orientation-deg 45
```

Output lists the FoV, the generated crop size, and its top aerial matches. The default rotation is 0° so runs are repeatable.

## Docker

The image is CPU-only on purpose: it is easy for a teaching assistant to build and run, and does not need a CUDA/GPU setup.

```bash
docker build -t congeo-fov-demo .
docker run --rm congeo-fov-demo
```

To run your local images without rebuilding, mount `sample/`:

```bash
docker run --rm -v "${PWD}/sample:/app/sample" congeo-fov-demo
```

On Windows PowerShell, replace `${PWD}` with `${PWD.Path}` if needed.

## Notes for the report

- ConGeo aims to improve robustness across ground-view orientation and FoV variations using a contrastive objective integrated with CVGL base architectures.
- This assignment visualises the **evaluation-side FoV variation** only. The official code documents 70°, 90°, 180°, and 360° evaluation configurations.
- Use your own images or images with permission to distribute. Large datasets and checkpoints should not be committed to this repository.

## References

- Li Mi et al., [*ConGeo: Robust Cross-view Geo-localization across Ground View Variations*](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2187_ECCV_2024_paper.php), ECCV 2024.
- [Official ConGeo implementation](https://github.com/eceo-epfl/ConGeo).

