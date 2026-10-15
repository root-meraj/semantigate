"""FFmpeg encoding and final audio/video assembly."""

from pathlib import Path
import shutil
import subprocess

import numpy as np
import psutil

from .config import RecorderConfig


def subprocess_options() -> dict:
    # Keep FFmpeg's console hidden on Windows, including GUI launches.
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}


def kill_process_tree(process) -> None:
    """FFmpeg may be behind a launcher shim; release its child file handles too."""
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for child in descendants:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)
    psutil.wait_procs(descendants, timeout=5)


def locate_ffmpeg(command: str) -> str:
    executable = shutil.which(command)
    if executable is None:
        raise RuntimeError("FFmpeg was not found. Install it or pass --ffmpeg with its executable path.")
    result = subprocess.run(
        [executable, "-hide_banner", "-encoders"], capture_output=True, text=True,
        timeout=15, **subprocess_options(),
    )
    if result.returncode or "libx264" not in result.stdout:
        raise RuntimeError("This FFmpeg build needs the libx264 H.264 encoder.")
    return executable


class H264Writer:
    def __init__(self, executable: str, path: Path, width: int, height: int, config: RecorderConfig):
        self.width, self.height = width, height
        self.frames = 0
        self.log_path = path.with_suffix(".ffmpeg.log")
        self._log = self.log_path.open("wb")
        command = [
            executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
            "-framerate", str(config.fps), "-i", "pipe:0", "-an",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
            "-preset", config.preset, "-crf", str(config.crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ]
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=self._log, **subprocess_options(),
            )
        except BaseException:
            self._log.close()
            raise

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError("Frame dimensions or dtype changed while recording.")
        if self.process.poll() is not None:
            raise RuntimeError(f"FFmpeg stopped unexpectedly. See {self.log_path}.")
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"FFmpeg could not encode a frame. See {self.log_path}.") from exc
        self.frames += 1

    def close(self) -> None:
        try:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                code = self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.abort()
                raise RuntimeError("FFmpeg did not finish within 30 seconds.")
            if code:
                raise RuntimeError(f"FFmpeg failed to finalize the video. See {self.log_path}.")
        finally:
            self._log.close()

    def abort(self) -> None:
        try:
            self.process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            # EOF can finalize the partial recording for recovery. Kill the
            # entire process tree only if it fails to shut down promptly.
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                kill_process_tree(self.process)
        finally:
            self._log.close()


def mux_recording(executable: str, video: Path, tracks: list[Path], duration: float, output: Path) -> None:
    """Copy H.264; pad/trim audio to the video timeline and encode it as AAC."""
    command = [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(video)]
    for track in tracks:
        command += ["-i", str(track)]
    if tracks:
        filters = []
        for index in range(1, len(tracks) + 1):
            filters.append(
                f"[{index}:a]aresample=48000,aformat=channel_layouts=stereo,"
                f"apad,atrim=duration={duration:.9f},asetpts=PTS-STARTPTS[a{index}]"
            )
        labels = "".join(f"[a{index}]" for index in range(1, len(tracks) + 1))
        filters.append(f"{labels}amix=inputs={len(tracks)}:duration=longest:normalize=1[audio]")
        command += ["-filter_complex", ";".join(filters), "-map", "0:v:0", "-map", "[audio]",
                    "-c:a", "aac", "-b:a", "160k"]
    else:
        command += ["-map", "0:v:0", "-an"]
    command += ["-c:v", "copy", "-t", f"{duration:.9f}", "-movflags", "+faststart", str(output)]
    log_path = output.with_suffix(".ffmpeg.log")
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log, **subprocess_options())
        try:
            code = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            kill_process_tree(process)
            raise RuntimeError(f"Audio/video assembly timed out. See {log_path}.")
    if code:
        raise RuntimeError(f"Audio/video assembly failed. See {log_path}.")
