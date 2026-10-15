from dataclasses import dataclass
from pathlib import Path
import math


@dataclass(frozen=True)
class RecorderConfig:
    output: Path
    fps: float = 10.0
    crf: int = 23
    preset: str = "veryfast"
    monitor: int | None = None  # None selects the primary monitor.
    audio: str = "system"
    system_device: int | None = None
    microphone_device: int | None = None
    changed_pixels: int = 1000
    pixel_threshold: int = 15
    refresh_seconds: float = 2.0
    duration: float | None = None
    gate_enabled: bool = True
    ffmpeg: str = "ffmpeg"

    def validate(self) -> None:
        if self.output.suffix.lower() != ".mp4":
            raise ValueError("Choose an output filename ending in .mp4.")
        if not math.isfinite(self.fps) or not 1 <= self.fps <= 60:
            raise ValueError("FPS must be between 1 and 60.")
        if not 0 <= self.crf <= 51:
            raise ValueError("CRF must be between 0 and 51 (lower is higher quality).")
        if self.preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium"}:
            raise ValueError("Unsupported H.264 preset.")
        if self.audio not in {"none", "system", "mic", "both"}:
            raise ValueError("Audio must be none, system, mic, or both.")
        if self.monitor is not None and self.monitor < 1:
            raise ValueError("Monitor indices start at 1.")
        if not 0 <= self.changed_pixels < 640 * 360:
            raise ValueError("Changed-pixel threshold must be between 0 and 230399.")
        if not 0 <= self.pixel_threshold <= 255:
            raise ValueError("Pixel intensity threshold must be between 0 and 255.")
        if not math.isfinite(self.refresh_seconds) or self.refresh_seconds <= 0:
            raise ValueError("Refresh interval must be positive.")
        if self.duration is not None and (not math.isfinite(self.duration) or self.duration <= 0):
            raise ValueError("Duration must be a positive number of seconds.")


def frames_for_duration(seconds: float, fps: float) -> int:
    """Cover elapsed time; quantization error is less than one video frame."""
    return max(1, math.ceil(max(0.0, seconds) * fps - 1e-9))
