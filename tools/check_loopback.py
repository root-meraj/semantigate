"""Verify real Windows loopback with a short tone and a clocked test picture.

Run from the project root: python -m tools.check_loopback
This plays a quiet 0.6-second tone and uses synthetic video, not your desktop.
"""

from datetime import datetime
import csv
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import numpy as np

from semantic_recorder.audio import AudioCapture
from semantic_recorder.config import RecorderConfig
from semantic_recorder.engine import Recorder
from semantic_recorder.video import subprocess_options


def main():
    if sys.platform != "win32":
        raise SystemExit("This check needs Windows WASAPI loopback.")
    output = Path("recordings") / f"loopback_check_{datetime.now():%Y%m%d_%H%M%S_%f}.mp4"
    trigger = threading.Event()
    clock = {"tone_at": None, "phase": 0, "remaining": 0}

    class ClockedAudio(AudioCapture):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            try:
                device = self._audio.get_default_wasapi_device(d_out=True)
                self.output_rate = int(device["defaultSampleRate"])
                self.output_channels = min(2, device["maxOutputChannels"])
                # Reuse the capture backend; independently initializing another
                # PortAudio instance can fail while WASAPI streams are active.
                self.output_stream = self._audio.open(
                    format=self._backend.paInt16, channels=self.output_channels,
                    rate=self.output_rate, output=True, output_device_index=device["index"],
                    frames_per_buffer=1024, stream_callback=self.play, start=False)
            except Exception:
                super().close(0)
                raise

        def play(self, _, count, timing, flags):
            samples = np.zeros((count, self.output_channels), dtype="<i2")
            if trigger.is_set() and clock["tone_at"] is None:
                clock["tone_at"] = (time.perf_counter() + timing["output_buffer_dac_time"]
                                    - timing["current_time"])
                clock["remaining"] = round(self.output_rate * 0.6)
            size = min(count, clock["remaining"])
            if size:
                tone = (np.sin(2 * np.pi * 440 * (np.arange(size) + clock["phase"])
                               / self.output_rate) * 4000).astype("<i2")
                samples[:size] = tone[:, None]
                clock["phase"] += size
                clock["remaining"] -= size
            return samples.tobytes(), self._backend.paContinue

        def start(self, origin):
            super().start(origin)
            self.output_stream.start_stream()

        def close(self, duration):
            try:
                if hasattr(self, "output_stream") and not self._closed:
                    if self.output_stream.is_active():
                        self.output_stream.stop_stream()
                    self.output_stream.close()
            finally:
                super().close(duration)

    class TestPicture:
        width, height = 320, 180

        def __enter__(self):
            return self

        def grab(self):
            bright = clock["tone_at"] is not None and time.perf_counter() >= clock["tone_at"]
            return np.full((180, 320, 3), 220 if bright else 30, dtype=np.uint8)

        def __exit__(self, *_):
            pass

    recorder = Recorder(RecorderConfig(output, audio="system", duration=3), TestPicture)
    errors = []

    def record():
        try:
            recorder.run()
        except Exception as exc:
            errors.append(str(exc))

    print("Playing a quiet 0.6-second tone while recording a synthetic picture.", flush=True)
    with patch("semantic_recorder.engine.AudioCapture", ClockedAudio):
        worker = threading.Thread(target=record, daemon=True)
        worker.start()
        deadline = time.monotonic() + 15
        while worker.is_alive() and recorder.snapshot()["elapsed"] < 0.6 and time.monotonic() < deadline:
            time.sleep(0.01)
        if worker.is_alive() and recorder.snapshot()["elapsed"] >= 0.6:
            trigger.set()
        else:
            recorder.stop()
        worker.join(timeout=20)
        if worker.is_alive():
            recorder.stop()
            raise SystemExit("Recorder did not finish promptly.")
        if errors:
            raise SystemExit("\n".join(errors))

    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:a:0",
                             "-ac", "1", "-ar", "48000", "-f", "s16le", "pipe:1"],
                            capture_output=True, check=True, **subprocess_options())
    captured = np.frombuffer(result.stdout, dtype="<i2").astype(float)
    energies = [np.sqrt(np.mean(captured[start:start + 480] ** 2))
                for start in range(0, len(captured) - 480, 480)]
    peak = max(energies, default=0)
    audible = [index * 0.01 for index, value in enumerate(energies) if value > max(20, peak * 0.2)]
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    print(f"Output: {output.resolve()}")
    print(f"Captured chunks: {summary['audio'][0]['chunks']}; peak RMS: {peak:.1f}")
    if not audible:
        raise SystemExit("No tone detected. Check the Windows speaker volume and selected output device.")
    with output.with_suffix(".metrics.csv").open(newline="") as metrics:
        rows = list(csv.DictReader(metrics))
    visual = next((float(row["elapsed_seconds"]) for row in rows
                   if row["state"] == "ACTIVE" and float(row["elapsed_seconds"]) > 0.4), None)
    if visual is None:
        raise SystemExit("The clocked visual marker did not appear in the recording.")
    difference = audible[0] - visual
    print(f"Tone onset: {audible[0]:.3f}s; visual marker: {visual:.3f}s; difference: {difference:.3f}s")
    if abs(difference) > 0.25:
        raise SystemExit("Tone was captured, but this device exceeded the check's 250 ms timing tolerance.")
    print("PASS: real loopback captured the tone within the 250 ms timing tolerance.")


if __name__ == "__main__":
    main()
