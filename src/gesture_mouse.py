"""Control your OS mouse with hand gestures, via ydotool. Trackpad-style:
the cursor follows your index finger's *movement*, not its absolute
position in frame -- move your hand out of frame and back (or pinch) to
reposition without dragging the cursor, same clutch behavior as lifting
your finger off a physical trackpad.

  - Point your index finger and move it: moves the cursor
  - Pinch thumb + index: left click -- hold and move to drag
  - Pinch thumb + middle: right click
  - Hold thumb + ring: scroll down (repeats while held)
  - Hold thumb + pinky: scroll up (repeats while held)

Cursor movement freezes while a right-click or scroll pinch is held (so
they don't drag the cursor along with them), but *not* during a left
pinch -- holding thumb+index and moving is exactly how you drag.

Fast hand motion gets an extra speed-based boost on top of the base
sensitivity (a real "throw" moves the cursor further than a slow drag
covering the same distance would), and briefly keeps the cursor moving
on last-known velocity if tracking drops the hand mid-swipe -- motion
blur on a fast flick is exactly when that's likely to happen.

Starts DISARMED -- press 'm' to arm/disarm. While armed, moving your hand
really moves your cursor and pinching really clicks, so arm it over an
empty part of the screen first. 'm' works every frame regardless of arm
state, so it's always your kill switch.

One-time setup this depends on (Wayland blocks synthetic mouse input by
default, so a direct Python mouse library can't do this alone):
  sudo apt install -y ydotool
  sudo usermod -aG input $USER
  # then log out/in, or `newgrp input` in the terminal you'll run this from
ydotoold normally auto-starts as a systemd --user service after install;
if cursor moves don't do anything, try:
  systemctl --user restart ydotool.service
"""

import argparse
import subprocess
import time
import tkinter as tk

import cv2
import numpy as np
from pynput.mouse import Controller as MouseReader

from hand_utils import HandTracker
from stream_server import StreamServer

SENSITIVITY = 3.5  # screen px moved per camera-frame px the fingertip moves, at rest (no speed boost)
CLICK_THRESHOLD = 0.35
SCROLL_THRESHOLD = 0.35
SCROLL_REPEAT_S = 0.15  # seconds between repeated scroll ticks while a scroll pinch is held
MOVE_DEADZONE_PX = 1.5  # ignore sub-pixel jitter in the raw camera-space delta before scaling by sensitivity

ACCEL_GAIN = 0.0012  # extra sensitivity multiplier per camera-px/sec of measured fingertip speed
MAX_ACCEL_MULT = 4.0  # cap on how much a fast throw can multiply the base sensitivity
THROW_MIN_SPEED = 1200.0  # camera-px/sec above which a dropout is treated as "mid-throw" and extrapolated
BLIND_EXTRAPOLATE_S = 0.2  # keep flying the cursor on last-known velocity for this long after losing the hand
VELOCITY_DECAY = 0.8  # per-frame decay applied to the extrapolated velocity while blind


def screen_size() -> tuple[int, int]:
    root = tk.Tk()
    root.withdraw()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return w, h


def ydotool(*args: str) -> None:
    subprocess.run(["ydotool", *args], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--screen-x0", type=int, default=0)
    parser.add_argument("--screen-y0", type=int, default=0)
    parser.add_argument("--screen-width", type=int, default=None)
    parser.add_argument("--screen-height", type=int, default=None)
    parser.add_argument("--sensitivity", type=float, default=SENSITIVITY)
    parser.add_argument("--click-threshold", type=float, default=CLICK_THRESHOLD)
    parser.add_argument("--scroll-threshold", type=float, default=SCROLL_THRESHOLD)
    parser.add_argument(
        "--serve", type=int, default=None,
        help="Serve the feed over HTTP on this port (MJPEG + key relay) instead of a native "
        "window -- used by the Phase 1 web launcher, not needed for direct CLI use.",
    )
    args = parser.parse_args()

    full_w, full_h = screen_size()
    screen_w = args.screen_width or full_w
    screen_h = args.screen_height or full_h
    screen_x0, screen_y0 = args.screen_x0, args.screen_y0
    screen_x1, screen_y1 = screen_x0 + screen_w, screen_y0 + screen_h
    if args.screen_width is None and full_w > 3000:
        print(
            f"Detected a {full_w}x{full_h} virtual screen (looks like multiple monitors "
            f"combined) -- without bounds the cursor can drift onto any of them. Pass "
            f"--screen-x0/--screen-width etc. to keep it clamped to one."
        )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    tracker = HandTracker(max_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    reader = MouseReader()  # read-only: pynput can read the real cursor position even though it can't move it
    stream = StreamServer(args.serve) if args.serve else None
    if stream:
        stream.start()

    armed = False
    prev_finger: np.ndarray | None = None  # None means "just reacquired" -> skip one frame of delta (clutch)
    prev_time: float | None = None
    last_velocity = np.zeros(2)  # camera px/sec, for the speed-boost and blind-extrapolation-through-dropout
    lost_since: float | None = None
    was_left = was_right = False
    scroll_last_tick_at = 0.0
    pinch_d = {"index": float("inf"), "middle": float("inf"), "ring": float("inf"), "pinky": float("inf")}
    frame_count = 0
    fps = 0.0
    fps_t0 = time.time()

    print(__doc__)
    print("Starting DISARMED. Press 'm' to arm, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        hands = tracker.process(frame)
        hand = hands[0] if hands else None

        is_left = is_right = is_scroll_down = is_scroll_up = freeze_move = False
        now_wall = time.time()

        if hand is not None:
            lost_since = None
            HandTracker.draw_landmarks(frame, hand)
            finger = hand.tip("index")
            ix, iy = finger

            pinch_d = {f: hand.pinch_distance(f) for f in ("index", "middle", "ring", "pinky")}
            is_left = pinch_d["index"] < args.click_threshold
            is_right = pinch_d["middle"] < args.click_threshold
            is_scroll_down = pinch_d["ring"] < args.scroll_threshold
            is_scroll_up = pinch_d["pinky"] < args.scroll_threshold
            # Left is excluded on purpose -- holding thumb+index and moving is a drag.
            freeze_move = is_right or is_scroll_down or is_scroll_up

            for fname, active in (
                ("index", is_left), ("middle", is_right), ("ring", is_scroll_down), ("pinky", is_scroll_up),
            ):
                pt = tuple(hand.tip(fname).astype(int))
                cv2.circle(frame, pt, 10, (0, 255, 120) if active else (0, 165, 255), 2, cv2.LINE_AA)

            if armed:
                # prev_finger still gets updated below regardless of freeze_move, so
                # releasing a right-click/scroll pinch doesn't cause a compensating jump.
                if prev_finger is not None and prev_time is not None and not freeze_move:
                    dt = max(now_wall - prev_time, 1e-3)
                    raw_dx, raw_dy = ix - prev_finger[0], iy - prev_finger[1]
                    if abs(raw_dx) < MOVE_DEADZONE_PX:
                        raw_dx = 0.0
                    if abs(raw_dy) < MOVE_DEADZONE_PX:
                        raw_dy = 0.0
                    last_velocity = np.array([raw_dx, raw_dy]) / dt
                    speed = float(np.hypot(*last_velocity))
                    accel_mult = float(np.clip(1.0 + ACCEL_GAIN * speed, 1.0, MAX_ACCEL_MULT))
                    dx = round(raw_dx * args.sensitivity * accel_mult)
                    dy = round(raw_dy * args.sensitivity * accel_mult)
                    if dx or dy:
                        cur_x, cur_y = reader.position
                        clamped_x = float(np.clip(cur_x + dx, screen_x0, screen_x1 - 1))
                        clamped_y = float(np.clip(cur_y + dy, screen_y0, screen_y1 - 1))
                        ydotool("mousemove", "-x", str(round(clamped_x - cur_x)), "-y", str(round(clamped_y - cur_y)))

                if is_left and not was_left:
                    ydotool("click", "0x40")  # left down -- a quick pinch-release is a click, holding+moving is a drag
                if not is_left and was_left:
                    ydotool("click", "0x80")  # left up
                if is_right and not was_right:
                    ydotool("click", "0xC1")  # right click stays atomic, no right-drag gesture requested

                if is_scroll_down or is_scroll_up:
                    if now_wall - scroll_last_tick_at >= SCROLL_REPEAT_S:
                        ydotool("mousemove", "-w", "-x", "0", "-y", "1" if is_scroll_down else "-1")
                        scroll_last_tick_at = now_wall
                else:
                    scroll_last_tick_at = 0.0  # next hold ticks immediately instead of waiting out a stale interval

            prev_finger = finger
            prev_time = now_wall
        else:
            if lost_since is None:
                lost_since = now_wall
            speed = float(np.hypot(*last_velocity))
            if (
                armed
                and not freeze_move
                and speed > THROW_MIN_SPEED
                and now_wall - lost_since < BLIND_EXTRAPOLATE_S
                and prev_time is not None
            ):
                # Fast motion blur is exactly when the tracker is most likely to lose the
                # fingertip mid-swipe -- keep flying the cursor on the last known velocity,
                # decaying it each frame, instead of the throw just stopping dead.
                dt = max(now_wall - prev_time, 1e-3)
                accel_mult = float(np.clip(1.0 + ACCEL_GAIN * speed, 1.0, MAX_ACCEL_MULT))
                dx = round(last_velocity[0] * dt * args.sensitivity * accel_mult)
                dy = round(last_velocity[1] * dt * args.sensitivity * accel_mult)
                if dx or dy:
                    cur_x, cur_y = reader.position
                    clamped_x = float(np.clip(cur_x + dx, screen_x0, screen_x1 - 1))
                    clamped_y = float(np.clip(cur_y + dy, screen_y0, screen_y1 - 1))
                    ydotool("mousemove", "-x", str(round(clamped_x - cur_x)), "-y", str(round(clamped_y - cur_y)))
                last_velocity = last_velocity * VELOCITY_DECAY
            else:
                last_velocity = np.zeros(2)

            prev_finger = None  # hand left frame -- next reacquire starts a fresh clutch, no jump
            prev_time = now_wall
            if armed and was_left:
                ydotool("click", "0x80")  # release a drag if tracking drops the hand mid-drag, avoid a stuck button

        was_left, was_right = is_left, is_right

        frame_count += 1
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10 / (now - fps_t0)
            fps_t0 = now

        status = "ARMED" if armed else "disarmed"
        color = (0, 0, 255) if armed else (150, 150, 150)
        cv2.putText(frame, f"{status}  fps: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(
            frame,
            f"pinch dist  index:{pinch_d['index']:.2f}(<{args.click_threshold:.2f})  "
            f"middle:{pinch_d['middle']:.2f}(<{args.click_threshold:.2f})  "
            f"ring:{pinch_d['ring']:.2f}(<{args.scroll_threshold:.2f})  "
            f"pinky:{pinch_d['pinky']:.2f}(<{args.scroll_threshold:.2f})",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
        )
        cv2.putText(
            frame,
            "m: arm/disarm  q: quit  |  thumb+index=L(drag)  thumb+middle=R  thumb+ring=scroll dn  thumb+pinky=scroll up",
            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )
        if stream:
            stream.publish(frame)
            key = stream.poll_key() or -1
        else:
            cv2.imshow("Gesture mouse", frame)
            key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            if armed and was_left:
                ydotool("click", "0x80")  # don't leave the button stuck down on exit
            break
        if key == ord("m"):
            if armed and was_left:
                ydotool("click", "0x80")  # don't leave the button stuck down on disarm
            armed = not armed
            print("ARMED" if armed else "disarmed")

    cap.release()
    if stream:
        stream.stop()
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
