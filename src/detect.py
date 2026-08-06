"""Combined live detection demo: multi-hand landmarks, body pose skeleton,
face mesh (eyes/mouth/outline), and face recognition -- all in one feed.

No portal/AR rendering here on purpose -- this is Phase 1, proving out
detection quality (multi-hand, full body, face) before building any effect
on top of it. See portal.py for the (separate, single-hand) circle-tracing
demo.

Running all four detectors every frame is CPU-heavy; use --no-<name> to
drop ones you don't need right now and watch the on-screen fps change.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from face_utils import FaceTracker
from hand_utils import HandTracker
from pose_utils import PoseTracker
from stream_server import StreamServer

_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
HAND_HOLD_S = 0.25  # keep drawing the last-seen hand(s) briefly through a dropout instead of flickering


def normalize_lighting(frame_bgr: np.ndarray) -> np.ndarray:
    """Local contrast boost (CLAHE on L*a*b* lightness).

    Helps the landmark models hold up in dim or unevenly lit rooms -- it
    does not make detection lighting-proof; very dark or blown-out frames
    still lose accuracy, there's no way around that with a single RGB
    camera.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--max-faces", type=int, default=2)
    parser.add_argument("--max-poses", type=int, default=1)
    parser.add_argument("--hands", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--face", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recognize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enhance-lighting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gallery", default="gallery/embeddings.pkl")
    parser.add_argument(
        "--hand-confidence", type=float, default=0.5,
        help="Lower = tracks smaller/farther hands longer, at the cost of more false positives.",
    )
    parser.add_argument(
        "--det-size", type=int, default=320,
        help="InsightFace detector input size (recognize.py's default is 640, "
        "which is the single heaviest thing running here per frame).",
    )
    parser.add_argument(
        "--serve", type=int, default=None,
        help="Serve the feed over HTTP on this port (MJPEG + key relay) instead of a native "
        "window -- used by the Phase 1 web launcher, not needed for direct CLI use.",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    stream = StreamServer(args.serve) if args.serve else None
    if stream:
        stream.start()

    hand_tracker = (
        HandTracker(
            max_hands=args.max_hands,
            min_detection_confidence=args.hand_confidence,
            min_presence_confidence=args.hand_confidence,
            min_tracking_confidence=args.hand_confidence,
        )
        if args.hands
        else None
    )
    pose_tracker = PoseTracker(max_poses=args.max_poses) if args.pose else None
    face_tracker = FaceTracker(max_faces=args.max_faces) if args.face else None

    recognizer = None
    gallery: dict = {}
    face_id = None
    if args.recognize:
        gallery_path = Path(args.gallery)
        if gallery_path.exists():
            import recognize as face_id  # local: only pulls in insightface/onnxruntime if actually used
            gallery = face_id.load_gallery(gallery_path)
            recognizer = face_id.build_app(det_size=(args.det_size, args.det_size))
        else:
            print(f"No gallery at {gallery_path} -- run enroll.py first. Skipping face recognition.")

    print("Detection-only phase: hands, pose, face mesh, face recognition. Press 'q' to quit.")

    frame_count = 0
    fps = 0.0
    fps_t0 = time.time()
    held_hands: list = []
    held_hands_at = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        detect_frame = normalize_lighting(frame) if args.enhance_lighting else frame
        now_wall = time.time()

        n_hands = n_poses = n_faces = 0
        recognized: list[str] = []

        if hand_tracker:
            hands = hand_tracker.process(detect_frame)
            n_hands = len(hands)  # ground truth this frame, independent of the hold below
            if hands:
                held_hands, held_hands_at = hands, now_wall
            elif now_wall - held_hands_at > HAND_HOLD_S:
                held_hands = []
            # Briefly re-drawing the last-seen hand through a one-off dropout (motion
            # blur, a small/distant hand losing detection for a frame) reads as
            # continuous tracking instead of flicker -- it's display smoothing only,
            # not a claim the hand is still really there past HAND_HOLD_S.
            for hand in held_hands:
                HandTracker.draw_landmarks(frame, hand)

        if pose_tracker:
            poses = pose_tracker.process(detect_frame)
            n_poses = len(poses)
            for pose in poses:
                PoseTracker.draw_landmarks(frame, pose)

        if face_tracker:
            faces = face_tracker.process(detect_frame)
            n_faces = len(faces)
            for face in faces:
                FaceTracker.draw_landmarks(frame, face)

        if recognizer is not None:
            for f in recognizer.get(detect_frame):
                name, score = face_id.best_match(f.normed_embedding, gallery)
                recognized.append(name)
                x1, y1, x2, y2 = f.bbox.astype(int)
                color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, f"{name} ({score:.2f})", (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

        frame_count += 1
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10 / (now - fps_t0)
            fps_t0 = now

        debug = f"fps: {fps:.1f}  hands: {n_hands}  poses: {n_poses}  faces: {n_faces}"
        if recognized:
            debug += "  id: " + ", ".join(recognized)
        cv2.putText(frame, debug, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if stream:
            stream.publish(frame)
            key = stream.poll_key() or -1
        else:
            cv2.imshow("Detection", frame)
            key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    if stream:
        stream.stop()
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
