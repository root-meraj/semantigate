from dataclasses import asdict
from datetime import datetime
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

import cv2
import mss
import numpy as np
import psutil

from .audio import AudioCapture
from .config import RecorderConfig, frames_for_duration
from .gate import MotionGate
from .video import H264Writer, locate_ffmpeg, mux_recording


def screen_capture():
    return mss.MSS() if hasattr(mss, "MSS") else mss.mss()


def list_monitors() -> list[dict]:
    with screen_capture() as capture:
        return [{"index": index, **monitor} for index, monitor in enumerate(capture.monitors) if index]


class ScreenSource:
    def __init__(self, monitor=None):
        self.index = monitor

    def __enter__(self):
        self.capture = screen_capture()
        try:
            monitors = self.capture.monitors
            index = self.index
            if index is None:
                index = next((i for i, item in enumerate(monitors) if i and item.get("is_primary")), 1)
            if index >= len(monitors):
                raise ValueError(f"Monitor {index} is unavailable. Run --list-devices to see monitors.")
            self.monitor = dict(monitors[index])
            self.width, self.height = self.monitor["width"], self.monitor["height"]
            return self
        except BaseException:
            self.capture.close()
            raise

    def grab(self):
        return cv2.cvtColor(np.asarray(self.capture.grab(self.monitor)), cv2.COLOR_BGRA2BGR)

    def __exit__(self, *_):
        self.capture.close()


def publish_file(source: Path, destination: Path) -> None:
    """Publish completed artifacts without replacing an existing file."""
    if os.name == "nt":
        source.rename(destination)  # Windows rename fails if destination exists.
    else:
        os.link(source, destination)
        source.unlink()


class Recorder:
    def __init__(self, config: RecorderConfig, source_factory=None):
        self.config = config
        self._source_factory = source_factory or (lambda: ScreenSource(config.monitor))
        self._stop = threading.Event()
        self._stop_time = None
        self._lock = threading.Lock()
        self._preview = None
        self._running = False
        self.stats = {"status": "Ready", "elapsed": 0.0, "frames": 0, "captured": 0,
                      "active": 0, "idle": 0, "refresh": 0, "repeated": 0,
                      "late_frames": 0, "changed_pixels": 0, "error": None,
                      "output": None, "warnings": []}

    def stop(self):
        with self._lock:
            if self._stop_time is None:
                self._stop_time = time.perf_counter()
        self._stop.set()

    def snapshot(self):
        with self._lock:
            return {**self.stats, "warnings": list(self.stats["warnings"])}

    def preview(self):
        with self._lock:
            return None if self._preview is None else self._preview.copy()

    def _update(self, **values):
        with self._lock:
            self.stats.update(values)

    def run(self) -> dict:
        if self._running:
            raise RuntimeError("A Recorder instance can only run once.")
        self._running = True
        config = self.config
        work = None
        writer = audio = None
        origin = None
        self._update(status="Preparing")
        try:
            config.validate()
            output = config.output.expanduser().resolve()
            metrics_path = output.with_suffix(".metrics.csv")
            summary_path = output.with_suffix(".summary.json")
            for path in (output, metrics_path, summary_path):
                if path.exists():
                    raise FileExistsError(f"{path} already exists. Choose a new recording filename.")
            executable = locate_ffmpeg(config.ffmpeg)
            output.parent.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix=f".{output.stem}-", dir=output.parent))
            gate = MotionGate(config.changed_pixels, config.pixel_threshold,
                              config.refresh_seconds, config.gate_enabled)
            process = psutil.Process()
            process.cpu_percent()  # Prime non-blocking CPU measurements.
            cpu_percent = 0.0
            last_telemetry = last_preview = -1.0
            started_at = datetime.now().astimezone().isoformat()

            with self._source_factory() as source, (work / "metrics.csv").open("w", newline="") as metrics:
                log = csv.writer(metrics)
                log.writerow(["elapsed_seconds", "state", "changed_pixels", "captured_frames",
                              "encoded_frames", "repeated_frames", "late_frames",
                              "process_cpu_percent", "process_rss_mb", "system_ram_percent"])
                first_frame = source.grab()
                writer = H264Writer(executable, work / "video.mp4", source.width, source.height, config)
                audio = AudioCapture(config.audio, work, config.system_device, config.microphone_device)
                origin = time.perf_counter()
                audio.start(origin)
                frame = first_frame
                captured = active = idle = refresh = repeated = late_frames = 0
                held = None

                while True:
                    elapsed = time.perf_counter() - origin
                    if self._stop.is_set() or (config.duration is not None and elapsed >= config.duration):
                        break
                    # A slow capture/encoder leaves vacant timeline slots. Fill
                    # them with the PREVIOUS displayed frame, then show the new
                    # capture in the slot matching its capture time.
                    slot = int(elapsed * config.fps)
                    if held is not None:
                        while writer.frames < slot:
                            writer.write(held)
                            repeated += 1
                            late_frames += 1
                    decision = gate.select(frame, elapsed)
                    held = decision.frame
                    writer.write(held)
                    captured += 1
                    active += decision.state == "ACTIVE"
                    idle += decision.state == "IDLE"
                    refresh += decision.state == "REFRESH"
                    repeated += not decision.updated
                    if elapsed - last_telemetry >= 1.0:
                        cpu_percent = process.cpu_percent()
                        last_telemetry = elapsed
                    log.writerow([f"{elapsed:.6f}", decision.state, decision.changed_pixels, captured,
                                  writer.frames, repeated, late_frames, round(cpu_percent, 2),
                                  round(process.memory_info().rss / 1024 ** 2, 2),
                                  psutil.virtual_memory().percent])
                    if elapsed - last_preview >= 0.2:
                        scale = min(1.0, 640 / source.width, 360 / source.height)
                        preview = cv2.resize(held, (max(1, round(source.width * scale)),
                                                  max(1, round(source.height * scale))))
                        with self._lock:
                            self._preview = preview
                        last_preview = elapsed
                    self._update(status=decision.state, elapsed=elapsed, frames=writer.frames,
                                 captured=captured, active=active, idle=idle, refresh=refresh,
                                 repeated=repeated, late_frames=late_frames, changed_pixels=decision.changed_pixels)
                    audio.check()
                    deadline = origin + writer.frames / config.fps
                    if config.duration is not None:
                        deadline = min(deadline, origin + config.duration)
                    if self._stop.wait(max(0.0, deadline - time.perf_counter())):
                        break
                    if config.duration is not None and time.perf_counter() - origin >= config.duration:
                        break
                    frame = source.grab()

                ended = self._stop_time if self._stop_time is not None else time.perf_counter()
                elapsed = max(0.0, ended - origin)
                if config.duration is not None:
                    elapsed = min(elapsed, config.duration)
                if held is None:
                    # A Stop during setup still produces a valid one-frame MP4.
                    held = first_frame
                    writer.write(held)
                    captured = active = 1
                target = frames_for_duration(elapsed, config.fps)
                self._update(status="Finalizing", elapsed=elapsed)
                # Stop audio before filling the final video slots or muxing.
                audio.close(elapsed)
                errors = [track.error for track in audio.tracks if track.error]
                if errors:
                    raise RuntimeError("; ".join(errors))
                while writer.frames < target:
                    writer.write(held)
                    repeated += 1
                writer.close()
                frames = writer.frames
                writer = None

            duration = frames / config.fps
            mux_recording(executable, work / "video.mp4", [track.path for track in audio.tracks],
                          duration, work / "completed.mp4")
            warnings = []
            for track in audio.tracks:
                if track.buffer_flags or track.discontinuities or track.timestamp_fallbacks:
                    warnings.append(f"{track.kind} audio needed timing/buffer handling; inspect the summary.")
                if not track.chunks:
                    warnings.append(f"No PCM arrived from {track.kind}; its track contains silence.")
            settings = asdict(config)
            settings["output"] = str(output)
            summary = {"started_at": started_at, "settings": settings, "wall_clock_seconds": elapsed,
                       "video_seconds": duration, "encoded_frames": frames, "captured_frames": captured,
                       "active_samples": active, "idle_samples": idle, "refresh_samples": refresh,
                       "repeated_frames": repeated, "late_frames": late_frames,
                       "frame_reuse_percent": round(100 * repeated / frames, 2),
                       "output_bytes": (work / "completed.mp4").stat().st_size,
                       "audio": audio.metadata(), "warnings": warnings}
            (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            publish_file(work / "completed.mp4", output)
            try:
                publish_file(work / "metrics.csv", metrics_path)
                publish_file(work / "summary.json", summary_path)
                # Verify the exact temporary target before recursively cleaning
                # it. It must remain a direct child of the chosen output folder.
                if work.resolve().parent != output.parent or not work.name.startswith(f".{output.stem}-"):
                    raise OSError("Temporary directory no longer resolves inside the output folder.")
                shutil.rmtree(work)
            except OSError as exc:
                warnings.append(f"Video saved; sidecars or temporary files remain at {work}: {exc}")
            self._update(status="Saved", elapsed=elapsed, frames=frames, captured=captured, active=active,
                         idle=idle, refresh=refresh, repeated=repeated, late_frames=late_frames,
                         output=str(output), warnings=warnings)
            return summary
        except Exception as exc:
            message = str(exc)
            if work is not None:
                message += f"\nRecovery files and FFmpeg logs: {work}"
            self._update(status="Failed", error=message)
            raise RuntimeError(message) from exc
        finally:
            if audio is not None:
                audio.close(0.0 if origin is None else max(0.0, time.perf_counter() - origin))
            if writer is not None:
                writer.abort()
