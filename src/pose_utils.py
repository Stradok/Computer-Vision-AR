"""Body pose landmark tracking (MediaPipe Tasks PoseLandmarker).

Same Tasks-API pattern as hand_utils.py -- see that file's docstring for
why (mediapipe>=1.0 dropped mp.solutions).
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Torso + limbs only (indices 11-32). The pose model's own 0-10 facial
# points are coarse and redundant with FaceLandmarker's much richer mesh,
# so they're left undrawn here.
_POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


@dataclass
class Pose:
    landmarks: np.ndarray  # (33, 2) pixel coordinates
    visibility: np.ndarray  # (33,) per-landmark visibility score, 0..1


class PoseTracker:
    def __init__(self, max_poses: int = 1, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose landmark model not found at {model_path}.\n"
                f"Download it with:\n  curl -L -o \"{model_path}\" {MODEL_URL}"
            )
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=max_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray) -> list[Pose]:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        poses: list[Pose] = []
        for lm_list in result.pose_landmarks:
            pts = np.array([[p.x * w, p.y * h] for p in lm_list], dtype=np.float32)
            vis = np.array([p.visibility for p in lm_list], dtype=np.float32)
            poses.append(Pose(landmarks=pts, visibility=vis))
        return poses

    @staticmethod
    def draw_landmarks(frame_bgr: np.ndarray, pose: Pose, visibility_threshold: float = 0.5) -> None:
        for a, b in _POSE_CONNECTIONS:
            if pose.visibility[a] < visibility_threshold or pose.visibility[b] < visibility_threshold:
                continue
            pa = tuple(pose.landmarks[a].astype(int))
            pb = tuple(pose.landmarks[b].astype(int))
            cv2.line(frame_bgr, pa, pb, (0, 200, 200), 2)
        for i, (x, y) in enumerate(pose.landmarks.astype(int)):
            if pose.visibility[i] < visibility_threshold:
                continue
            cv2.circle(frame_bgr, (x, y), 3, (255, 140, 0), -1)
