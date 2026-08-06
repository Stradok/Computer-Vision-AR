"""Build a face gallery from reference photos.

Expected layout:
    data/known_faces/<person_name>/*.jpg

Enroll 2-3 photos per person with different looks (glasses on/off, beard
grown/shaved, etc.) — matching takes the best score across all of a
person's stored embeddings, so varied references make recognition more
robust to appearance changes.
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def build_app() -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def embed_image(app: FaceAnalysis, image_path: Path) -> np.ndarray | None:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  skip (unreadable): {image_path.name}")
        return None
    faces = app.get(img)
    if not faces:
        print(f"  skip (no face found): {image_path.name}")
        return None
    # In case a reference photo has bystanders in frame, use the largest face.
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.normed_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/known_faces")
    parser.add_argument("--out", default="gallery/embeddings.pkl")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    people = sorted(p for p in data_dir.iterdir() if p.is_dir()) if data_dir.exists() else []
    if not people:
        raise SystemExit(
            f"No person folders found in {data_dir}. Expected data/known_faces/<name>/*.jpg"
        )

    app = build_app()
    gallery: dict[str, list[np.ndarray]] = {}

    for person_dir in people:
        name = person_dir.name
        images = sorted({p for ext in IMAGE_EXTENSIONS for p in person_dir.glob(ext)})
        if not images:
            print(f"{name}: no images found, skipping")
            continue
        print(f"{name}: enrolling {len(images)} image(s)")
        embeddings = [e for e in (embed_image(app, p) for p in images) if e is not None]
        if embeddings:
            gallery[name] = embeddings
            print(f"  -> {len(embeddings)} usable embedding(s)")
        else:
            print(f"  -> no usable faces, {name} not enrolled")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(gallery, f)
    print(f"\nSaved gallery with {len(gallery)} people to {out_path}")


if __name__ == "__main__":
    main()
