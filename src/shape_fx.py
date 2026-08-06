"""Filter effects rendered inside an arbitrary polygon mask.

Same masked-compositing idea as portal_fx.py's circle, generalized to any
simple polygon (cv2.fillPoly instead of a distance-from-center test).
"""

import cv2
import numpy as np

FILTER_NAMES = ["spiral", "invert", "pixelate", "edges"]


def _polygon_mask(frame_shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(frame_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points.astype(np.int32)], 255)
    return mask


def _composite(frame: np.ndarray, filtered: np.ndarray, mask: np.ndarray) -> None:
    mask3 = cv2.merge([mask, mask, mask])
    np.copyto(frame, filtered, where=mask3 > 0)


def render_spiral(frame: np.ndarray, points: np.ndarray, t: float) -> None:
    h, w = frame.shape[:2]
    x0, y0 = max(int(points[:, 0].min()), 0), max(int(points[:, 1].min()), 0)
    x1, y1 = min(int(points[:, 0].max()) + 1, w), min(int(points[:, 1].max()) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return
    cx, cy = points[:, 0].mean(), points[:, 1].mean()
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    spiral = np.sin(angle * 6 + dist * 0.15 - t * 4.0)
    pattern = ((spiral + 1) * 0.5 * 255).astype(np.uint8)
    colored = cv2.applyColorMap(pattern, cv2.COLORMAP_INFERNO)

    mask = _polygon_mask(frame.shape[:2], points)[y0:y1, x0:x1]
    roi = frame[y0:y1, x0:x1]
    blended = cv2.addWeighted(roi, 0.25, colored, 0.9, 0)
    _composite(roi, blended, mask)


def render_invert(frame: np.ndarray, points: np.ndarray, t: float) -> None:
    mask = _polygon_mask(frame.shape[:2], points)
    inverted = cv2.bitwise_not(frame)
    _composite(frame, inverted, mask)


def render_pixelate(frame: np.ndarray, points: np.ndarray, t: float, block: int = 14) -> None:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(w // block, 1), max(h // block, 1)), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = _polygon_mask(frame.shape[:2], points)
    _composite(frame, mosaic, mask)


def render_edges(frame: np.ndarray, points: np.ndarray, t: float) -> None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    colored = cv2.applyColorMap(edges, cv2.COLORMAP_SPRING)
    colored[edges == 0] = frame[edges == 0]
    mask = _polygon_mask(frame.shape[:2], points)
    _composite(frame, colored, mask)


_RENDERERS = {
    "spiral": render_spiral,
    "invert": render_invert,
    "pixelate": render_pixelate,
    "edges": render_edges,
}


def render_filter(frame: np.ndarray, points: np.ndarray, t: float, name: str) -> None:
    """points: (N, 2) polygon vertices in placement order, N >= 3."""
    _RENDERERS[name](frame, points, t)
    cv2.polylines(frame, [points.astype(np.int32)], True, (255, 210, 120), 2, cv2.LINE_AA)
