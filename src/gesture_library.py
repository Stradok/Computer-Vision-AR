"""Recorded hand-shape gestures mapped to actions (directional cursor pushes, mute/unmute,
...), persisted to disk so they survive across runs. Recording itself happens in
gesture_mouse.py's main loop; this module just owns storage + nearest-match lookup.
"""

import json
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "models" / "gestures.json"


class GestureLibrary:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path
        self.templates: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        self.templates = {action: np.array(vec, dtype=np.float32) for action, vec in data.items()}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {action: vec.tolist() for action, vec in self.templates.items()}
        self.path.write_text(json.dumps(data, indent=2))

    def set(self, action: str, vector: np.ndarray) -> None:
        self.templates[action] = vector.astype(np.float32)
        self._save()

    def clear(self, action: str) -> None:
        if action in self.templates:
            del self.templates[action]
            self._save()

    def match(self, vector: np.ndarray, threshold: float) -> tuple[str, float] | None:
        """Nearest recorded template by RMSE over the normalized-shape vector, or None if
        nothing recorded is within `threshold`.
        """
        best_action, best_dist = None, float("inf")
        for action, template in self.templates.items():
            dist = float(np.sqrt(np.mean((vector - template) ** 2)))
            if dist < best_dist:
                best_action, best_dist = action, dist
        if best_action is not None and best_dist < threshold:
            return best_action, best_dist
        return None
