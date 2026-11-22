"""Control your OS mouse with hand gestures, via ydotool. Two-handed:

  - One hand only moves the cursor. Its *shape* decides what happens:
      - A shape you've recorded as a directional gesture (accel_up/down/
        left/right) pushes the cursor continuously in that fixed screen
        direction at a flat speed for as long as you hold it -- a deliberate
        stand-in for the old "tilt the finger to drag" trick, which kept
        misfiring because a pointing hand held at a slight angle looks a lot
        like a pointing hand held straight. A recorded shape has to be
        genuinely different from your other poses, not just rotated, so it
        doesn't trigger by accident.
      - A shape recorded for a one-shot action (mute/unmute) fires that
        action once, on the frame it's first recognized -- not repeatedly
        while held.
      - Otherwise, falls back to built-in poses: pointing (only index
        extended) = accelerated mode for covering distance, fast motion gets
        an extra speed-based boost. Open palm (fingers spread) = precise
        mode: raw fingertip movement at a flat sensitivity, no boost.
    A fist, a relaxed hand, or any other in-between shape is a hard rest:
    zero movement regardless of where its landmarks land. This is a real
    pose check, not just "moving less", so a resting hand can't get its
    geometry misread as pointing in some direction and drift on its own.
    While tracking (either built-in mode), it's trackpad-style: the cursor
    follows the fingertip's *movement*, not its absolute position in frame,
    so moving your hand out of frame and back repositions without dragging
    the cursor (clutch behavior) -- and dropping into a resting shape and
    back out does the same, so pausing to rest never causes a jump when you
    resume.
  - The other hand only does clicks/scroll, via pinches against its thumb:
      - Pinch thumb + index tip: left click -- hold and move the *cursor*
        hand to drag.
      - Pinch thumb + middle: right click
      - Hold thumb + ring: scroll down (repeats while held)
      - Hold thumb + pinky: scroll up (repeats while held)
    Cursor movement is entirely decoupled from this hand's pinches, so
    nothing needs to freeze to avoid click-jitter.

Gestures are recorded from the Phase 1 web launcher, not from this script
directly: the page has Record buttons for each action next to the
sensitivity slider. Clicking one gives you ~1.2s to get your cursor hand
into position, then ~0.6s where it samples your held shape and saves it to
models/gestures.json (shared across runs). While a recording is in
progress, cursor movement is frozen so positioning your hand can't drag
anything around underneath it.

By default your right hand moves the cursor and your left hand clicks --
press 'h' to swap which physical hand does which if that's backwards (hand
labels are classified on the mirrored camera feed, so which is which can
be counterintuitive). The HUD shows which role each tracked hand currently
has.

When run via the Phase 1 web launcher (--serve), a sensitivity slider in the
page overrides --sensitivity live.

Fast hand motion gets an extra speed-based boost on top of the base
sensitivity (a real "throw" moves the cursor further than a slow drag
covering the same distance would), and briefly keeps the cursor moving
on last-known velocity if tracking drops the cursor hand mid-swipe --
motion blur on a fast flick is exactly when that's likely to happen.

Starts DISARMED -- press 'm' to arm/disarm. While armed, moving your cursor
hand really moves your cursor and pinching your click hand really clicks,
so arm it with both hands clear of anything important first. 'm' works
every frame regardless of arm state, so it's always your kill switch.

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

from gesture_library import GestureLibrary
from hand_utils import HandTracker
from stream_server import StreamServer

SENSITIVITY = 5.0  # screen px moved per camera-frame px the fingertip moves, at rest (no speed boost)
SENSITIVITY_MIN = 1.0  # slider range floor -- keep in sync with phase1/public/js/main.js
SENSITIVITY_MAX = 12.0  # slider range ceiling -- keep in sync with phase1/public/js/main.js
CLICK_THRESHOLD = 0.35
SCROLL_THRESHOLD = 0.35
SCROLL_REPEAT_S = 0.15  # seconds between repeated scroll ticks while a scroll pinch is held
MOVE_DEADZONE_PX = 1.5  # ignore sub-pixel jitter in the raw camera-space delta before scaling by sensitivity

ACCEL_GAIN = 0.0025  # extra sensitivity multiplier per camera-px/sec of measured fingertip speed
MAX_ACCEL_MULT = 6.0  # cap on how much a fast throw can multiply the base sensitivity
THROW_MIN_SPEED = 800.0  # camera-px/sec above which a dropout is treated as "mid-throw" and extrapolated
BLIND_EXTRAPOLATE_S = 0.35  # keep flying the cursor on last-known velocity for this long after losing the hand
VELOCITY_DECAY = 0.85  # per-frame decay applied to the extrapolated velocity while blind

# Recorded gestures: a hand shape recorded via the web UI is matched against the live hand's
# normalized_shape() by RMSE; the closest recorded template under this threshold wins over
# the built-in pointing/open_palm poses.
GESTURE_MATCH_RMSE = 0.22  # normalized-shape RMSE below which a recorded gesture counts as a match
GESTURE_PUSH_SPEED = 700.0  # camera-px/sec-equivalent continuous cursor push while a directional gesture is held
RECORD_PREP_S = 1.2  # grace period after a record request before sampling starts, to get the hand in position
RECORD_CAPTURE_S = 0.6  # how long the pose is sampled once capture starts

GESTURE_DIRECTIONS = {
    "accel_up": (0.0, -1.0),
    "accel_down": (0.0, 1.0),
    "accel_left": (-1.0, 0.0),
    "accel_right": (1.0, 0.0),
}
GESTURE_ONE_SHOT = {"mute", "unmute"}  # fire once on the frame first recognized, not every frame held


def screen_size() -> tuple[int, int]:
    root = tk.Tk()
    root.withdraw()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return w, h


def ydotool(*args: str) -> None:
    subprocess.run(["ydotool", *args], check=False)


def set_system_mute(muted: bool) -> None:
    subprocess.run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "1" if muted else "0"], check=False)


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

    tracker = HandTracker(max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    reader = MouseReader()  # read-only: pynput can read the real cursor position even though it can't move it
    stream = StreamServer(args.serve) if args.serve else None
    if stream:
        stream.start()

    armed = False
    swapped = False  # 'h' toggles which physical hand (by MediaPipe's Left/Right label) does which job
    prev_finger: np.ndarray | None = None  # None means "just reacquired" -> skip one frame of delta (clutch)
    prev_time: float | None = None
    last_velocity = np.zeros(2)  # camera px/sec, for the speed-boost and blind-extrapolation-through-dropout
    last_accel_enabled = True  # whether the mode active right before a dropout allows the accel boost
    lost_since: float | None = None
    gestures = GestureLibrary()
    pending_record: str | None = None  # action name currently being recorded, from a web-UI /record request
    record_started_at: float | None = None
    record_samples: list[np.ndarray] = []
    active_gesture: str | None = None  # matched action last frame, for one-shot (mute/unmute) edge detection
    was_left = was_right = False
    scroll_last_tick_at = 0.0
    pinch_d = {"index": float("inf"), "middle": float("inf"), "ring": float("inf"), "pinky": float("inf")}
    frame_count = 0
    fps = 0.0
    fps_t0 = time.time()

    print(__doc__)
    print("Starting DISARMED. Press 'm' to arm, 'h' to swap hands, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        hands = tracker.process(frame)
        for hand in hands:
            HandTracker.draw_landmarks(frame, hand)

        cursor_label = "Left" if swapped else "Right"
        click_label = "Right" if swapped else "Left"
        cursor_hand = next((hd for hd in hands if hd.handedness == cursor_label), None)
        click_hand = next((hd for hd in hands if hd.handedness == click_label), None)

        is_left = is_right = is_scroll_down = is_scroll_up = False
        now_wall = time.time()
        sensitivity = stream.get_param("sensitivity", args.sensitivity) if stream else args.sensitivity

        if stream:
            gesture_req = stream.poll_gesture_request()
            if gesture_req is not None:
                req_action, req_clear = gesture_req
                if req_clear:
                    gestures.clear(req_action)
                    print(f"Cleared gesture '{req_action}'")
                else:
                    pending_record = req_action
                    record_started_at = now_wall
                    record_samples = []

        # ---- cursor hand: movement only, never gated by the click hand's pinches ----
        cursor_pose = "none"
        is_tracking = False
        gesture_active: str | None = None
        gesture_dist: float | None = None
        recording = pending_record is not None

        if recording:
            # Frozen: positioning the hand for a recording must never drag the real cursor
            # around underneath it.
            prev_finger = None
            prev_time = now_wall
            last_velocity = np.zeros(2)
            elapsed = now_wall - record_started_at
            if elapsed < RECORD_PREP_S:
                cv2.putText(
                    frame, f"Get ready: '{pending_record}'  {RECORD_PREP_S - elapsed:.1f}s",
                    (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
                )
            else:
                if cursor_hand is not None:
                    lost_since = None
                    record_samples.append(cursor_hand.normalized_shape())
                    cv2.circle(frame, tuple(cursor_hand.tip("index").astype(int)), 14, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    f"Hold '{pending_record}' steady...  "
                    f"{max(RECORD_PREP_S + RECORD_CAPTURE_S - elapsed, 0.0):.1f}s",
                    (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
            if elapsed >= RECORD_PREP_S + RECORD_CAPTURE_S:
                if record_samples:
                    gestures.set(pending_record, np.mean(np.stack(record_samples), axis=0))
                    print(f"Recorded gesture '{pending_record}' ({len(record_samples)} samples)")
                else:
                    print(f"Recording '{pending_record}' failed -- no hand detected, try again")
                pending_record = None
                record_samples = []
        elif cursor_hand is not None:
            lost_since = None
            finger = cursor_hand.tip("index")
            ix, iy = finger
            match = gestures.match(cursor_hand.normalized_shape(), GESTURE_MATCH_RMSE)
            if match is not None:
                gesture_active, gesture_dist = match

            if gesture_active in GESTURE_DIRECTIONS:
                # A recorded directional shape: fixed continuous push in that screen
                # direction for as long as the shape is held, independent of hand
                # translation -- the deliberate replacement for the old tilt-joystick.
                is_tracking = True
                dx_dir, dy_dir = GESTURE_DIRECTIONS[gesture_active]
                cv2.circle(frame, tuple(finger.astype(int)), 12, (0, 140, 255), 2, cv2.LINE_AA)
                if armed:
                    dt = max(now_wall - prev_time, 1e-3) if prev_time is not None else 1.0 / 30.0
                    dx = round(dx_dir * GESTURE_PUSH_SPEED * dt * sensitivity)
                    dy = round(dy_dir * GESTURE_PUSH_SPEED * dt * sensitivity)
                    if dx or dy:
                        cur_x, cur_y = reader.position
                        clamped_x = float(np.clip(cur_x + dx, screen_x0, screen_x1 - 1))
                        clamped_y = float(np.clip(cur_y + dy, screen_y0, screen_y1 - 1))
                        ydotool("mousemove", "-x", str(round(clamped_x - cur_x)), "-y", str(round(clamped_y - cur_y)))
                prev_finger = None  # fixed push, not delta-tracked -- translate mode resumes fresh on release
                prev_time = now_wall
                last_velocity = np.zeros(2)
                last_accel_enabled = False
            elif gesture_active in GESTURE_ONE_SHOT:
                cv2.circle(frame, tuple(finger.astype(int)), 12, (255, 0, 200), 2, cv2.LINE_AA)
                if armed and active_gesture != gesture_active:
                    set_system_mute(gesture_active == "mute")
                    print(f"Gesture '{gesture_active}' triggered")
                prev_finger = None
                prev_time = now_wall
                last_velocity = np.zeros(2)
            else:
                cursor_pose = cursor_hand.pose()
                # Pointing (only index extended) or an open palm (hand held up, fingers
                # spread) both read as "actively steering the cursor" -- anything else
                # (fist, a relaxed hand, an ambiguous in-between shape) is a hard rest.
                is_tracking = cursor_pose in ("pointing", "open_palm")

                if is_tracking:
                    # Pointing is the fast/accelerated mode (speed boost), for covering
                    # distance. Open palm is the precise mode: raw fingertip movement at a
                    # flat sensitivity, no boost -- for fine placement.
                    accel_enabled = cursor_pose == "pointing"
                    last_accel_enabled = accel_enabled  # carries into extrapolation if this hand then drops out
                    color = (255, 200, 0) if accel_enabled else (0, 220, 0)
                    cv2.circle(frame, tuple(finger.astype(int)), 12, color, 2, cv2.LINE_AA)

                    if armed and prev_finger is not None and prev_time is not None:
                        dt = max(now_wall - prev_time, 1e-3)
                        raw_dx, raw_dy = ix - prev_finger[0], iy - prev_finger[1]
                        if abs(raw_dx) < MOVE_DEADZONE_PX:
                            raw_dx = 0.0
                        if abs(raw_dy) < MOVE_DEADZONE_PX:
                            raw_dy = 0.0
                        last_velocity = np.array([raw_dx, raw_dy]) / dt
                        speed = float(np.hypot(*last_velocity))
                        accel_mult = (
                            float(np.clip(1.0 + ACCEL_GAIN * speed, 1.0, MAX_ACCEL_MULT)) if accel_enabled else 1.0
                        )
                        dx = round(raw_dx * sensitivity * accel_mult)
                        dy = round(raw_dy * sensitivity * accel_mult)
                        if dx or dy:
                            cur_x, cur_y = reader.position
                            clamped_x = float(np.clip(cur_x + dx, screen_x0, screen_x1 - 1))
                            clamped_y = float(np.clip(cur_y + dy, screen_y0, screen_y1 - 1))
                            ydotool(
                                "mousemove", "-x", str(round(clamped_x - cur_x)), "-y", str(round(clamped_y - cur_y)),
                            )

                    prev_finger = finger
                    prev_time = now_wall
                else:
                    # A fist or ambiguous in-between shape is a hard rest: no movement.
                    # Without this gate, a resting hand's geometry could get misread as one
                    # of the active modes and the cursor just keeps sliding even though the
                    # hand isn't doing anything recognizable -- rest must be a genuine pose
                    # change, not just "moving less."
                    cv2.circle(frame, tuple(finger.astype(int)), 12, (120, 120, 120), 2, cv2.LINE_AA)
                    prev_finger = None  # resuming tracking starts a fresh clutch, like lifting off a trackpad
                    prev_time = now_wall
                    last_velocity = np.zeros(2)  # don't let a stale throw carry into a later dropout extrapolation
            active_gesture = gesture_active
        else:
            active_gesture = None
            if lost_since is None:
                lost_since = now_wall
            speed = float(np.hypot(*last_velocity))
            if (
                armed
                and speed > THROW_MIN_SPEED
                and now_wall - lost_since < BLIND_EXTRAPOLATE_S
                and prev_time is not None
            ):
                # Fast motion blur is exactly when the tracker is most likely to lose the
                # fingertip mid-swipe -- keep flying the cursor on the last known velocity,
                # decaying it each frame, instead of the throw just stopping dead.
                dt = max(now_wall - prev_time, 1e-3)
                accel_mult = (
                    float(np.clip(1.0 + ACCEL_GAIN * speed, 1.0, MAX_ACCEL_MULT)) if last_accel_enabled else 1.0
                )
                dx = round(last_velocity[0] * dt * sensitivity * accel_mult)
                dy = round(last_velocity[1] * dt * sensitivity * accel_mult)
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

        # ---- click hand: all buttons + scroll, never moves the cursor ----
        if click_hand is not None:
            pinch_d = {f: click_hand.pinch_distance(f) for f in ("index", "middle", "ring", "pinky")}
            is_left = pinch_d["index"] < args.click_threshold
            is_right = pinch_d["middle"] < args.click_threshold
            is_scroll_down = pinch_d["ring"] < args.scroll_threshold
            is_scroll_up = pinch_d["pinky"] < args.scroll_threshold

            for fname, active in (
                ("index", is_left), ("middle", is_right), ("ring", is_scroll_down), ("pinky", is_scroll_up),
            ):
                pt = tuple(click_hand.tip(fname).astype(int))
                cv2.circle(frame, pt, 10, (0, 255, 120) if active else (0, 165, 255), 2, cv2.LINE_AA)

            if armed:
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
        elif armed and was_left:
            ydotool("click", "0x80")  # click hand vanished mid-drag -- release, don't leave the button stuck down

        was_left, was_right = is_left, is_right

        frame_count += 1
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10 / (now - fps_t0)
            fps_t0 = now

        status = "ARMED" if armed else "disarmed"
        color = (0, 0, 255) if armed else (150, 150, 150)
        cv2.putText(
            frame, f"{status}  fps: {fps:.1f}  sensitivity: {sensitivity:.1f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
        )
        if recording:
            cursor_status = "(recording)"
        elif cursor_hand is None:
            cursor_status = "(missing)"
        elif gesture_active:
            cursor_status = f"(gesture: {gesture_active})"
        elif is_tracking:
            cursor_status = f"({cursor_pose.upper()}, {'accel' if cursor_pose == 'pointing' else 'precise'})"
        else:
            cursor_status = f"(resting: {cursor_pose})"
        cv2.putText(
            frame,
            f"cursor hand: {cursor_label}{cursor_status}   "
            f"click hand: {click_label}{'(tracked)' if click_hand else '(missing)'}",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1,
        )
        gesture_info = (
            f"  gesture:{gesture_active}({gesture_dist:.2f}<{GESTURE_MATCH_RMSE:.2f})" if gesture_active else ""
        )
        cv2.putText(
            frame,
            f"pinch dist  index:{pinch_d['index']:.2f}(<{args.click_threshold:.2f})  "
            f"middle:{pinch_d['middle']:.2f}(<{args.click_threshold:.2f})  "
            f"ring:{pinch_d['ring']:.2f}(<{args.scroll_threshold:.2f})  "
            f"pinky:{pinch_d['pinky']:.2f}(<{args.scroll_threshold:.2f}){gesture_info}",
            (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
        )
        cv2.putText(
            frame,
            "m: arm/disarm  h: swap hands  q: quit  |  thumb+index=L(drag)  thumb+middle=R  "
            "thumb+ring=scroll dn  thumb+pinky=scroll up",
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
        if key == ord("h"):
            if armed and was_left:
                ydotool("click", "0x80")  # don't leave the button stuck down while roles swap under it
            swapped = not swapped
            print(f"Cursor hand: {'Left' if swapped else 'Right'}  Click hand: {'Right' if swapped else 'Left'}")

    cap.release()
    if stream:
        stream.stop()
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
