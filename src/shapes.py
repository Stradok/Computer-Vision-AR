"""Pinch (thumb + index finger) to drop a point in space. 3+ points close
into a shape with a filter effect rendered inside it. Works with either
hand, or both at once for more points faster.

Keys: 1-4 switch filter (spiral/invert/pixelate/edges), c clears all
points, q quits.
"""

import argparse

import cv2
import numpy as np

from hand_utils import HandTracker
from shape_fx import FILTER_NAMES, render_filter
from stream_server import StreamServer

MAX_POINTS = 8
PINCH_THRESHOLD = 0.35


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

    points: list[np.ndarray] = []
    was_pinching: dict[int, bool] = {}
    filter_idx = 0
    start_time = cv2.getTickCount()

    print(__doc__)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        t = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()

        hands = tracker.process(frame)
        for i, hand in enumerate(hands):
            pinching = hand.is_pinching(PINCH_THRESHOLD)
            mid = (hand.tip("thumb") + hand.tip("index")) / 2
            ring_color = (0, 255, 120) if pinching else (0, 165, 255)
            cv2.circle(frame, tuple(mid.astype(int)), 10, ring_color, 2, cv2.LINE_AA)

            if pinching and not was_pinching.get(i, False) and len(points) < MAX_POINTS:
                points.append(mid)
            was_pinching[i] = pinching

        if len(points) >= 3:
            render_filter(frame, np.array(points), t, FILTER_NAMES[filter_idx])
        elif len(points) >= 1:
            pts = np.array(points, dtype=np.int32)
            for p in pts:
                cv2.circle(frame, tuple(p), 5, (255, 210, 120), -1)
            if len(pts) >= 2:
                cv2.polylines(frame, [pts], False, (255, 210, 120), 2, cv2.LINE_AA)

        debug = f"filter: {FILTER_NAMES[filter_idx]}  points: {len(points)}/{MAX_POINTS}"
        cv2.putText(frame, debug, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(
            frame, "pinch to place point | 1-4 filter | c clear | q quit",
            (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        if stream:
            stream.publish(frame)
            key = stream.poll_key() or -1
        else:
            cv2.imshow("Shapes", frame)
            key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("c"):
            points.clear()
            was_pinching.clear()
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            filter_idx = key - ord("1")

    cap.release()
    if stream:
        stream.stop()
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
