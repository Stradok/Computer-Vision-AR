"""Doctor-Strange-style portal: trace a circle in the air with your index
finger to open it, make a fist to close it.

Requires models/hand_landmarker.task (see hand_utils.py / README.md).
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np

from hand_utils import Hand, HandTracker
from portal_fx import render_portal
from stream_server import StreamServer

TRAIL_MAXLEN = 400  # generous headroom so a slow trace never evicts its own starting point
MIN_TRAIL_POINTS_TO_CHECK = 20
CLOSURE_DIST_PX = 40
MIN_LOOP_SPAN_PX = 60
TRAIL_TIMEOUT_S = 4.0
FIST_DWELL_S = 0.6
OPEN_ANIM_S = 0.5
CLOSE_ANIM_S = 0.4
MAX_MISSED_FRAMES = 6  # ~0.2s of grace at 30fps for a dropped detection / pose flicker mid-trace


class PortalState:
    def __init__(self) -> None:
        self.phase = "idle"  # idle -> tracing -> opening -> open -> closing
        self.trail: deque[tuple[float, float]] = deque(maxlen=TRAIL_MAXLEN)
        self.trail_started_at = 0.0
        self.center = (0, 0)
        self.radius = 0.0
        self.anim_start = 0.0
        self.fist_since: float | None = None
        self.missed_frames = 0

    def update(self, hands: list[Hand], now: float) -> None:
        hand = hands[0] if hands else None

        if self.phase == "idle":
            if hand and hand.pose() == "pointing":
                self.phase = "tracing"
                self.trail.clear()
                self.trail_started_at = now
                self.missed_frames = 0
                self.trail.append(tuple(hand.tip("index")))

        elif self.phase == "tracing":
            good = hand is not None and hand.pose() == "pointing"
            if good:
                self.missed_frames = 0
            else:
                # Tolerate a short run of dropped detections / pose flicker
                # (motion blur on a fast trace, a single misclassified
                # frame) instead of nuking the whole trail on frame one.
                self.missed_frames += 1
                if self.missed_frames > MAX_MISSED_FRAMES:
                    self.phase = "idle"
                    self.trail.clear()
                    self.missed_frames = 0
                    return
            if now - self.trail_started_at > TRAIL_TIMEOUT_S:
                self.phase = "idle"
                self.trail.clear()
                self.missed_frames = 0
                return
            if not good:
                return
            self.trail.append(tuple(hand.tip("index")))
            loop = self._find_closed_loop()
            if loop is not None:
                self.center = tuple(loop.mean(axis=0).astype(int))
                self.radius = float(
                    max(loop[:, 0].max() - loop[:, 0].min(), loop[:, 1].max() - loop[:, 1].min()) / 2
                )
                self.anim_start = now
                self.phase = "opening"
                self.trail.clear()

        elif self.phase == "opening":
            if now - self.anim_start >= OPEN_ANIM_S:
                self.phase = "open"

        elif self.phase == "open":
            if hand and hand.pose() == "fist":
                self.fist_since = self.fist_since or now
                if now - self.fist_since >= FIST_DWELL_S:
                    self.anim_start = now
                    self.phase = "closing"
                    self.fist_since = None
            else:
                self.fist_since = None

        elif self.phase == "closing":
            if now - self.anim_start >= CLOSE_ANIM_S:
                self.phase = "idle"

    def _find_closed_loop(self) -> np.ndarray | None:
        """Return the sub-trail forming a just-closed loop, or None.

        Scans backward from the most recent point for the closest-in-time
        earlier point the fingertip has returned near -- rather than only
        comparing the trail's absolute first/last points, which breaks if
        the trail contains a long approach before the actual circle, or if
        the deque has evicted the true start of a slow trace.
        """
        if len(self.trail) < MIN_TRAIL_POINTS_TO_CHECK:
            return None
        pts = np.array(self.trail, dtype=np.float32)
        latest = pts[-1]
        for start_idx in range(len(pts) - MIN_TRAIL_POINTS_TO_CHECK, -1, -1):
            if float(np.linalg.norm(latest - pts[start_idx])) < CLOSURE_DIST_PX:
                loop = pts[start_idx:]
                span = max(loop[:, 0].max() - loop[:, 0].min(), loop[:, 1].max() - loop[:, 1].min())
                if span >= MIN_LOOP_SPAN_PX:
                    return loop
        return None

    def open_amount(self, now: float) -> float:
        if self.phase == "opening":
            return min((now - self.anim_start) / OPEN_ANIM_S, 1.0)
        if self.phase == "open":
            return 1.0
        if self.phase == "closing":
            return max(1.0 - (now - self.anim_start) / CLOSE_ANIM_S, 0.0)
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--show-landmarks", action="store_true")
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
    # Uncapped capture resolution (some webcams default to 1080p+) inflates
    # every per-frame cv2 op (flip, color convert, draw) well beyond what
    # the landmark model itself needs -- pin it down for consistent latency.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    tracker = HandTracker(max_hands=1)
    state = PortalState()
    stream = StreamServer(args.serve) if args.serve else None
    if stream:
        stream.start()

    print("Trace a circle in the air with your index finger to open the portal.")
    print("Make a fist to close it. Press 'q' to quit.")

    start_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        now = time.time() - start_time

        hands = tracker.process(frame)
        state.update(hands, now)

        for hand in hands:
            if args.show_landmarks:
                tracker.draw_landmarks(frame, hand)
            if state.phase == "tracing":
                cv2.circle(frame, tuple(hand.tip("index").astype(int)), 6, (0, 255, 255), -1)

        if state.phase == "tracing" and len(state.trail) > 1:
            pts = np.array(state.trail, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 255), 2)

        amount = state.open_amount(now)
        if amount > 0:
            render_portal(frame, state.center, state.radius, now, amount)

        pose_label = hands[0].pose() if hands else "no hand"
        debug_line = f"phase: {state.phase}  pose: {pose_label}  trail: {len(state.trail)}"
        cv2.putText(frame, debug_line, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if stream:
            stream.publish(frame)
            key = stream.poll_key() or -1
        else:
            cv2.imshow("Portal", frame)
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
