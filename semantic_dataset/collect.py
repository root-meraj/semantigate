from pathlib import Path
import math
import time

import cv2
import numpy as np
from PIL import Image

from .core import IMAGE_EXTENSIONS, SessionWriter


def validate_collection(count: int, interval: float, min_change: float) -> None:
    if count < 1:
        raise ValueError("Count must be positive; it limits capture attempts, not accepted unique images.")
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Interval must be positive.")
    if not math.isfinite(min_change) or not 0 <= min_change <= 255:
        raise ValueError("Minimum mean grayscale change must be between 0 and 255.")


class CandidateFilter:
    def __init__(self, minimum=0.5):
        self.minimum = minimum
        self.last = None

    def accept(self, image: Image.Image) -> bool:
        gray = np.asarray(image.convert("L").resize((160, 90)), dtype=np.int16)
        if self.last is not None and np.abs(gray - self.last).mean() < self.minimum:
            return False
        self.last = gray
        return True


def import_images(folder: Path, root: Path, session: str) -> dict:
    paths = sorted(path for path in folder.expanduser().resolve().iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError("No PNG/JPEG/BMP/WebP screenshots were found in that folder.")
    writer = SessionWriter(root, session, {"kind": "import", "folder": folder.name})
    saved = 0
    for path in paths:
        with Image.open(path) as image:
            saved += writer.add(image) is not None
    return {"attempts": len(paths), "saved": saved, "session": str(writer.directory)}


def extract_video(video: Path, root: Path, session: str, *, count=40, interval=3.0, min_change=0.5) -> dict:
    validate_collection(count, interval, min_change)
    video = video.expanduser().resolve()
    if not video.is_file():
        raise ValueError(f"Video file not found: {video}")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError("OpenCV could not open this video.")
        writer = SessionWriter(root, session, {"kind": "video", "filename": video.name,
                                               "interval_seconds": interval, "min_mean_change": min_change})
        selection = CandidateFilter(min_change)
        attempts = saved = 0
        for index in range(count):
            seconds = index * interval
            if not capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000):
                raise ValueError("This video decoder does not support seeking by time.")
            ok, frame = capture.read()
            if not ok:
                break
            attempts += 1
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if selection.accept(image):
                saved += writer.add(image, seconds) is not None
        return {"attempts": attempts, "saved": saved, "session": str(writer.directory)}
    finally:
        capture.release()


def capture_screen(root: Path, session: str, *, monitor=None, count=40, interval=3.0, min_change=0.5) -> dict:
    validate_collection(count, interval, min_change)
    if monitor is not None and monitor < 1:
        raise ValueError("Monitor indices start at 1.")
    from semantic_recorder.engine import ScreenSource
    with ScreenSource(monitor) as source:
        writer = SessionWriter(root, session, {"kind": "screen", "monitor": monitor,
                                               "interval_seconds": interval, "min_mean_change": min_change})
        selection = CandidateFilter(min_change)
        started = time.perf_counter()
        attempts = saved = 0
        try:
            for index in range(count):
                time.sleep(max(0.0, started + index * interval - time.perf_counter()))
                image = Image.fromarray(cv2.cvtColor(source.grab(), cv2.COLOR_BGR2RGB))
                attempts += 1
                if selection.accept(image):
                    saved += writer.add(image, time.perf_counter() - started) is not None
                print(f"Capture {attempts}/{count}: {saved} candidates saved", flush=True)
        except KeyboardInterrupt:
            print("Collection stopped; saved candidates and session metadata were kept.")
        return {"attempts": attempts, "saved": saved, "session": str(writer.directory)}
