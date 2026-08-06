"""Face mesh landmark tracking (MediaPipe Tasks FaceLandmarker) -- eyes,
lips, and face outline.

Same Tasks-API pattern as hand_utils.py. This is landmark tracking, not
identity -- for "who is this", see recognize.py (InsightFace), which is a
separate model/pipeline entirely.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# Google's published FaceMesh index subsets -- enough to draw eyes/mouth/
# outline without rendering the full 478-point tesselation. "left"/"right"
# are the subject's own left/right (mirrored in a selfie view), matching
# MediaPipe's convention, not screen-left/right.
_LIPS = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
]
_LEFT_EYE = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362),
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362),
]
_RIGHT_EYE = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133),
]
_FACE_OVAL = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454),
    (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400),
    (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54),
    (54, 103), (103, 67), (67, 109), (109, 10),
]


@dataclass
class Face:
    landmarks: np.ndarray  # (478, 2) pixel coords: 468 mesh points + 10 iris points


class FaceTracker:
    def __init__(self, max_faces: int = 1, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face landmark model not found at {model_path}.\n"
                f"Download it with:\n  curl -L -o \"{model_path}\" {MODEL_URL}"
            )
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray) -> list[Face]:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        faces: list[Face] = []
        for lm_list in result.face_landmarks:
            pts = np.array([[p.x * w, p.y * h] for p in lm_list], dtype=np.float32)
            faces.append(Face(landmarks=pts))
        return faces

    @staticmethod
    def draw_landmarks(frame_bgr: np.ndarray, face: Face) -> None:
        for a, b in _FACE_OVAL:
            pa = tuple(face.landmarks[a].astype(int))
            pb = tuple(face.landmarks[b].astype(int))
            cv2.line(frame_bgr, pa, pb, (180, 180, 180), 1, cv2.LINE_AA)
        for a, b in _LEFT_EYE + _RIGHT_EYE:
            pa = tuple(face.landmarks[a].astype(int))
            pb = tuple(face.landmarks[b].astype(int))
            cv2.line(frame_bgr, pa, pb, (255, 220, 0), 1, cv2.LINE_AA)
        for a, b in _LIPS:
            pa = tuple(face.landmarks[a].astype(int))
            pb = tuple(face.landmarks[b].astype(int))
            cv2.line(frame_bgr, pa, pb, (0, 120, 255), 1, cv2.LINE_AA)
