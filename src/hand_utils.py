"""Hand landmark tracking (MediaPipe Tasks HandLandmarker) and simple pose classification.

Uses the new MediaPipe Tasks API -- mediapipe>=1.0 dropped the old
`mp.solutions.hands` API in favor of this one, which requires a model
bundle on disk (see models/hand_landmarker.task).
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

_FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
_FINGER_PIPS = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _smoothing_factor(t_e: float, cutoff: np.ndarray | float) -> np.ndarray | float:
    r = 2 * np.pi * cutoff * t_e
    return r / (r + 1)


class _OneEuroFilter:
    """Per-landmark jitter filter (Casiez et al.) tuned for low lag on fast motion.

    Raw landmark coordinates are noisy enough frame-to-frame that pose
    classification (is_extended) and loop-closure detection both flicker.
    A plain moving average would smooth that out but lag behind a fast
    circle trace; One-Euro adapts its cutoff to the current velocity so a
    still hand gets heavy smoothing while a fast-moving fingertip stays
    responsive.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.4, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None
        self.t_prev: float | None = None

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x
        t_e = max(t - self.t_prev, 1e-3)
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


@dataclass
class Hand:
    landmarks: np.ndarray  # (21, 2) pixel coordinates
    handedness: str

    def tip(self, finger: str) -> np.ndarray:
        return self.landmarks[_FINGER_TIPS[finger]]

    def pinch_distance(self, finger: str = "index") -> float:
        """Thumb-tip-to-`finger`-tip distance, normalized by wrist-to-middle-MCP
        (landmark 9) hand size so the same threshold works whether the hand
        is close to the camera or far away.
        """
        thumb, tip = self.landmarks[4], self.landmarks[_FINGER_TIPS[finger]]
        hand_size = float(np.linalg.norm(self.landmarks[9] - self.landmarks[0]))
        if hand_size < 1e-6:
            return float("inf")
        return float(np.linalg.norm(thumb - tip)) / hand_size

    def is_pinching(self, threshold: float = 0.35, finger: str = "index") -> bool:
        return self.pinch_distance(finger) < threshold

    def is_extended(self, finger: str) -> bool:
        tip = self.landmarks[_FINGER_TIPS[finger]]
        pip = self.landmarks[_FINGER_PIPS[finger]]
        wrist = self.landmarks[0]
        # Tip farther from the wrist than the pip joint == extended.
        # Robust to hand rotation/tilt, unlike a fixed-axis comparison.
        return float(np.linalg.norm(tip - wrist)) > float(np.linalg.norm(pip - wrist))

    def pose(self) -> str:
        extended = {f: self.is_extended(f) for f in _FINGER_TIPS}
        # Thumb is excluded from the counts: its tip-vs-wrist distance
        # heuristic is unreliable (different kinematics from the other
        # fingers), so it flickers between extended/curled regardless of
        # actual pose and was causing "pointing" to drop out mid-trace.
        non_thumb_extended = sum(extended[f] for f in ("index", "middle", "ring", "pinky"))
        if extended["index"] and non_thumb_extended == 1:
            return "pointing"
        if non_thumb_extended == 0:
            return "fist"
        if non_thumb_extended >= 3:
            return "open_palm"
        return "other"


class HandTracker:
    def __init__(
        self,
        max_hands: int = 1,
        model_path: Path = MODEL_PATH,
        min_detection_confidence: float = 0.6,
        min_presence_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand landmark model not found at {model_path}.\n"
                f"Download it with:\n  curl -L -o \"{model_path}\" {MODEL_URL}"
            )
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        self._filters: list[_OneEuroFilter] = []

    def process(self, frame_bgr: np.ndarray) -> list[Hand]:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33  # detect_for_video requires monotonically increasing timestamps
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        t = self._timestamp_ms / 1000.0
        hands: list[Hand] = []
        for i, (lm_list, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
            pts = np.array([[p.x * w, p.y * h] for p in lm_list], dtype=np.float32)
            if i >= len(self._filters):
                self._filters.append(_OneEuroFilter())
            pts = self._filters[i](t, pts)
            label = handedness[0].category_name if handedness else "Unknown"
            hands.append(Hand(landmarks=pts, handedness=label))
        # Drop filter state for slots with no hand this frame, so a hand that
        # re-enters the frame starts fresh instead of smoothing toward stale
        # coordinates from before it left.
        del self._filters[len(result.hand_landmarks):]
        return hands

    @staticmethod
    def draw_landmarks(frame_bgr: np.ndarray, hand: Hand) -> None:
        for a, b in _HAND_CONNECTIONS:
            pa = tuple(hand.landmarks[a].astype(int))
            pb = tuple(hand.landmarks[b].astype(int))
            cv2.line(frame_bgr, pa, pb, (0, 200, 0), 2)
        for x, y in hand.landmarks.astype(int):
            cv2.circle(frame_bgr, (x, y), 3, (0, 120, 255), -1)
