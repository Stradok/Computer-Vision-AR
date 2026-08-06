"""Recognize enrolled faces from a webcam feed or a static image.

Run enroll.py first to build gallery/embeddings.pkl.
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from stream_server import StreamServer

# Cosine similarity threshold for buffalo_l's normalized embeddings.
# Raise it if you get false matches between different people, lower it
# if a known person is showing up as Unknown too often.
MATCH_THRESHOLD = 0.38


def build_app(det_size: tuple[int, int] = (640, 640)) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=det_size)
    return app


def load_gallery(path: Path) -> dict[str, list[np.ndarray]]:
    with open(path, "rb") as f:
        return pickle.load(f)


def best_match(embedding: np.ndarray, gallery: dict[str, list[np.ndarray]]) -> tuple[str, float]:
    best_name, best_score = "Unknown", -1.0
    for name, embeddings in gallery.items():
        score = max(float(np.dot(embedding, e)) for e in embeddings)
        if score > best_score:
            best_name, best_score = name, score
    if best_score < MATCH_THRESHOLD:
        return "Unknown", best_score
    return best_name, best_score


def annotate(frame, faces, gallery):
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        name, score = best_match(face.normed_embedding, gallery)
        color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} ({score:.2f})"
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def run_webcam(app: FaceAnalysis, gallery: dict, camera_index: int = 0, serve: int | None = None) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {camera_index}")
    stream = StreamServer(serve) if serve else None
    if stream:
        stream.start()
    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = app.get(frame)
        frame = annotate(frame, faces, gallery)

        if stream:
            stream.publish(frame)
            key = stream.poll_key() or -1
        else:
            cv2.imshow("Face recognition", frame)
            key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    if stream:
        stream.stop()
    else:
        cv2.destroyAllWindows()


def run_image(app: FaceAnalysis, gallery: dict, image_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Could not read {image_path}")
    faces = app.get(img)
    annotated = annotate(img, faces, gallery)
    out_path = image_path.with_name(image_path.stem + "_recognized" + image_path.suffix)
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved annotated image to {out_path}")
    for face in faces:
        name, score = best_match(face.normed_embedding, gallery)
        print(f"  {name}: {score:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", default="gallery/embeddings.pkl")
    parser.add_argument("--image", help="Path to a single image instead of using the webcam")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--serve", type=int, default=None,
        help="Serve the feed over HTTP on this port (MJPEG + key relay) instead of a native "
        "window -- used by the Phase 1 web launcher, not needed for direct CLI use.",
    )
    args = parser.parse_args()

    gallery_path = Path(args.gallery)
    if not gallery_path.exists():
        raise SystemExit(f"No gallery found at {gallery_path}. Run enroll.py first.")
    gallery = load_gallery(gallery_path)
    if not gallery:
        raise SystemExit("Gallery is empty. Enroll at least one person first.")

    app = build_app()

    if args.image:
        run_image(app, gallery, Path(args.image))
    else:
        run_webcam(app, gallery, args.camera, serve=args.serve)


if __name__ == "__main__":
    main()
