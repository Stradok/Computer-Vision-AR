"""Procedural Doctor-Strange-style portal visual: an animated rotating
spiral rendered into a circular mask, composited onto the video frame.
"""

import cv2
import numpy as np


def render_portal(
    frame: np.ndarray,
    center: tuple[int, int],
    radius: float,
    t: float,
    open_amount: float,
) -> np.ndarray:
    """Draw a portal onto `frame` in place and return it.

    open_amount: 0..1, animates the radius for opening/closing transitions.
    t: elapsed seconds, drives the spiral's rotation.
    """
    r = int(radius * max(open_amount, 0.0))
    if r <= 1:
        return frame

    cx, cy = center
    h, w = frame.shape[:2]
    x0, y0 = max(cx - r, 0), max(cy - r, 0)
    x1, y1 = min(cx + r, w), min(cy + r, h)
    if x1 <= x0 or y1 <= y0:
        return frame

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    spiral = np.sin(angle * 6 + dist * 0.15 - t * 4.0)
    pattern = ((spiral + 1) * 0.5 * 255).astype(np.uint8)
    colored = cv2.applyColorMap(pattern, cv2.COLORMAP_INFERNO)

    mask = (dist <= r).astype(np.uint8) * 255
    mask3 = cv2.merge([mask, mask, mask])

    roi = frame[y0:y1, x0:x1]
    blended = cv2.addWeighted(roi, 0.25, colored, 0.9, 0)
    frame[y0:y1, x0:x1] = np.where(mask3 > 0, blended, roi)

    cv2.circle(frame, center, r, (60, 140, 255), 3, lineType=cv2.LINE_AA)
    cv2.circle(frame, center, max(r - 6, 1), (255, 210, 120), 1, lineType=cv2.LINE_AA)
    return frame
