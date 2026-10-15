from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GateDecision:
    frame: np.ndarray
    state: str
    changed_pixels: int
    updated: bool


class MotionGate:
    """Compare against the displayed frame, so a new static slide isn't absorbed.

    A periodic refresh bounds how long small edits below the threshold can stay
    hidden. This is a pixel gate; it does not classify screen elements.
    """

    def __init__(self, changed_pixels=1000, pixel_threshold=15, refresh_seconds=2.0, enabled=True):
        self.changed_pixels = changed_pixels
        self.pixel_threshold = pixel_threshold
        self.refresh_seconds = refresh_seconds
        self.enabled = enabled
        self._reference = None
        self._held_frame = None
        self._last_update = 0.0

    def select(self, frame: np.ndarray, elapsed: float) -> GateDecision:
        small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        changed = 0
        if self._reference is not None:
            diff = cv2.absdiff(blur, self._reference)
            _, mask = cv2.threshold(diff, self.pixel_threshold, 255, cv2.THRESH_BINARY)
            mask = cv2.dilate(mask, None, iterations=2)
            changed = int(cv2.countNonZero(mask))

        if self._held_frame is None or not self.enabled:
            state = "ACTIVE"
        elif changed > self.changed_pixels:
            state = "ACTIVE"
        elif elapsed - self._last_update >= self.refresh_seconds:
            state = "REFRESH"
        else:
            return GateDecision(self._held_frame, "IDLE", changed, False)

        self._reference = blur
        self._held_frame = frame.copy()
        self._last_update = elapsed
        return GateDecision(self._held_frame, state, changed, True)
