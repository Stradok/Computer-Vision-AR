"""Two-hand "finger frame" filter game -- hold up both hands, thumb+index
tips forming a rectangle-ish frame between them (the classic director's
viewfinder gesture), and a filter effect fills the frame live as you move
it around. Same filters as shapes.py (shape_fx.py), applied to a
continuously-tracked frame instead of pinch-placed points.

Keys: 1-4 switch filter (spiral/invert/pixelate/edges), q quits.
"""

import argparse

import cv2
import numpy as np

from hand_utils import HandTracker
from shape_fx import FILTER_NAMES, render_filter
from stream_server import StreamServer

HOLD_S = 0.25  # keep the last frame briefly through a one-hand dropout instead of collapsing it


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
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

    tracker = HandTracker(max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    stream = StreamServer(args.serve) if args.serve else None
    if stream:
        stream.start()

    filter_idx = 0
    held_corners: np.ndarray | None = None
    held_at = 0.0
    start_time = cv2.getTickCount()

    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        t = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()

        hands = tracker.process(frame)
        for hand in hands:
            for finger in ("thumb", "index"):
                pt = tuple(hand.tip(finger).astype(int))
                cv2.circle(frame, pt, 6, (255, 210, 120), -1)

        if len(hands) == 2:
            corners = np.array(
                [hands[0].tip("thumb"), hands[0].tip("index"), hands[1].tip("thumb"), hands[1].tip("index")],
                dtype=np.float32,
            )
            hull = cv2.convexHull(corners)[:, 0, :]  # orders the perimeter regardless of hand rotation
            if len(hull) >= 3:
                held_corners, held_at = hull, t

        if held_corners is not None and t - held_at < HOLD_S and len(held_corners) >= 3:
            render_filter(frame, held_corners, t, FILTER_NAMES[filter_idx])

        debug = f"filter: {FILTER_NAMES[filter_idx]}  hands: {len(hands)}"
        cv2.putText(frame, debug, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(
            frame, "both hands, thumb+index forming a frame | 1-4 filter | q quit",
            (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        if stream:
            stream.publish(frame)
            key = stream.poll_key() or -1
        else:
            cv2.imshow("Finger Frame", frame)
            key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            filter_idx = key - ord("1")

    cap.release()
    if stream:
        stream.stop()
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
