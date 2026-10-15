import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import wave

import numpy as np

from semantic_recorder.audio import TimelineWaveWriter
from semantic_recorder.config import RecorderConfig, frames_for_duration
from semantic_recorder.engine import Recorder
from semantic_recorder.gate import MotionGate
from semantic_recorder.video import H264Writer, locate_ffmpeg, mux_recording, subprocess_options


class ConfigTests(unittest.TestCase):
    def test_frame_quantization(self):
        self.assertEqual(frames_for_duration(1.0, 10), 10)
        self.assertEqual(frames_for_duration(1.01, 10), 11)
        self.assertEqual(frames_for_duration(0, 10), 1)

    def test_reject_invalid_settings(self):
        for settings in ({"fps": 0}, {"fps": float("nan")}, {"duration": -1},
                         {"duration": float("inf")}, {"crf": 52}, {"monitor": 0},
                         {"audio": "invalid"}, {"refresh_seconds": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                RecorderConfig(Path("recording.mp4"), **settings).validate()


class GuiStateTests(unittest.TestCase):
    def test_completion_waits_for_worker_exit(self):
        import tkinter as tk
        from semantic_recorder.gui import RecorderApp
        from unittest.mock import Mock
        root = tk.Tk()
        root.withdraw()
        try:
            with patch("semantic_recorder.gui.list_monitors", return_value=[]), \
                 patch("semantic_recorder.gui.list_audio_devices", return_value=[]):
                app = RecorderApp(root)
            app.recorder = Mock()
            app.recorder.snapshot.return_value = {"status": "Saved", "output": "recording.mp4", "warnings": []}
            app.recorder.preview.return_value = None
            app.worker = Mock()
            app.worker.is_alive.return_value = True
            app._set_busy(True)
            app.poll()
            self.assertNotEqual(app._last_status, "Saved")
            app.worker.is_alive.return_value = False
            app.poll()
            self.assertEqual(app.status.get(), "Recording saved")
            self.assertEqual(str(app.start_button.cget("state")), "normal")
            self.assertEqual(str(app.stop_button.cget("state")), "disabled")
        finally:
            root.destroy()


class GateTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((360, 640, 3), 80, dtype=np.uint8)

    def test_static_frames_are_held(self):
        gate = MotionGate()
        first = gate.select(self.frame, 0)
        second = gate.select(self.frame.copy(), 0.1)
        self.assertEqual(first.state, "ACTIVE")
        self.assertEqual(second.state, "IDLE")
        self.assertIs(first.frame, second.frame)

    def test_new_slide_updates_once_then_becomes_idle(self):
        gate = MotionGate()
        gate.select(self.frame, 0)
        slide = np.full_like(self.frame, 220)
        self.assertEqual(gate.select(slide, 0.1).state, "ACTIVE")
        self.assertEqual(gate.select(slide, 0.2).state, "IDLE")

    def test_tiny_edit_is_visible_on_refresh(self):
        gate = MotionGate(refresh_seconds=2)
        gate.select(self.frame, 0)
        edited = self.frame.copy()
        edited[50:54, 50:54] = 255
        result = gate.select(edited, 0.1)
        self.assertFalse(result.updated)
        refreshed = gate.select(edited, 2.01)
        self.assertEqual(refreshed.state, "REFRESH")
        np.testing.assert_array_equal(refreshed.frame, edited)

    def test_sensor_noise_does_not_wake_gate(self):
        gate = MotionGate()
        gate.select(self.frame, 0)
        noise = np.random.default_rng(7).integers(-3, 4, size=self.frame.shape)
        noisy = (self.frame.astype(np.int16) + noise).astype(np.uint8)
        self.assertEqual(gate.select(noisy, 0.1).state, "IDLE")

    def test_no_gate_updates_every_capture(self):
        gate = MotionGate(enabled=False)
        gate.select(self.frame, 0)
        self.assertTrue(gate.select(self.frame, 0.1).updated)


class AudioTimelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "track.wav"

    def tearDown(self):
        self.directory.cleanup()

    def test_late_audio_starts_with_silence_and_ends_on_timeline(self):
        writer = TimelineWaveWriter(self.path, 1000, 1)
        signal = np.full(100, 1000, dtype="<i2").tobytes()
        writer.append(signal, 0.25)
        writer.close(1.0)
        with wave.open(str(self.path), "rb") as track:
            pcm = np.frombuffer(track.readframes(track.getnframes()), dtype="<i2")
        self.assertEqual(len(pcm), 1000)
        self.assertTrue(np.all(pcm[:250] == 0))
        self.assertTrue(np.all(pcm[250:350] == 1000))
        self.assertTrue(np.all(pcm[350:] == 0))

    def test_negative_start_trims_pre_recording_audio(self):
        writer = TimelineWaveWriter(self.path, 1000, 1)
        writer.append(np.arange(100, dtype="<i2").tobytes(), -0.05)
        writer.close(0.05)
        with wave.open(str(self.path), "rb") as track:
            pcm = np.frombuffer(track.readframes(track.getnframes()), dtype="<i2")
        np.testing.assert_array_equal(pcm, np.arange(50, 100))

    def test_discontinuity_keeps_later_sound_at_correct_time(self):
        writer = TimelineWaveWriter(self.path, 1000, 1)
        signal = np.full(100, 2000, dtype="<i2").tobytes()
        writer.append(signal, 0.0)
        writer.append(signal, 0.4)
        self.assertEqual(writer.discontinuities, 1)
        writer.close(0.5)
        with wave.open(str(self.path), "rb") as track:
            pcm = np.frombuffer(track.readframes(track.getnframes()), dtype="<i2")
        self.assertTrue(np.all(pcm[100:400] == 0))
        self.assertTrue(np.all(pcm[400:500] == 2000))

    def test_no_callbacks_produce_silent_track(self):
        writer = TimelineWaveWriter(self.path, 1000, 2)
        writer.close(0.4)
        with wave.open(str(self.path), "rb") as track:
            self.assertEqual(track.getnframes(), 400)
            self.assertEqual(track.readframes(400), bytes(400 * 4))


class SyntheticSource:
    width, height = 161, 91  # Exercise H.264's even-dimension padding.

    def __init__(self, delay=0.0, fail_after=None):
        self.delay = delay
        self.fail_after = fail_after
        self.count = 0

    def __enter__(self):
        return self

    def grab(self):
        if self.fail_after is not None and self.count >= self.fail_after:
            raise RuntimeError("Synthetic screen disconnected")
        if self.count:
            time.sleep(self.delay)
        self.count += 1
        return np.full((self.height, self.width, 3), 32 if self.count < 3 else 160, dtype=np.uint8)

    def __exit__(self, *_):
        pass


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe required")
class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.output = self.root / "recording.mp4"

    def tearDown(self):
        self.directory.cleanup()

    def probe(self, path):
        result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                 "-of", "json", str(path)], check=True, capture_output=True,
                                text=True, **subprocess_options())
        return json.loads(result.stdout)

    def test_fast_source_does_not_speed_up_timeline(self):
        recorder = Recorder(RecorderConfig(self.output, audio="none", duration=0.7), SyntheticSource)
        summary = recorder.run()
        info = self.probe(self.output)
        self.assertEqual(summary["encoded_frames"], 7)
        self.assertAlmostEqual(float(info["format"]["duration"]), 0.7, places=2)
        self.assertEqual(info["streams"][0]["codec_name"], "h264")
        self.assertEqual((info["streams"][0]["width"], info["streams"][0]["height"]), (162, 92))
        self.assertGreater(summary["idle_samples"], 0)
        self.assertTrue(self.output.with_suffix(".metrics.csv").exists())

    def test_slow_capture_preserves_duration_with_repeated_slots(self):
        recorder = Recorder(RecorderConfig(self.output, audio="none", duration=1.1),
                            lambda: SyntheticSource(delay=0.27))
        summary = recorder.run()
        self.assertEqual(summary["encoded_frames"], 11)
        self.assertGreater(summary["late_frames"], 0)
        self.assertLess(summary["captured_frames"], 11)
        self.assertAlmostEqual(float(self.probe(self.output)["format"]["duration"]), 1.1, places=2)

    def test_stop_request_during_capture_does_not_include_shutdown_time(self):
        recorder = Recorder(RecorderConfig(self.output, audio="none"), lambda: SyntheticSource(delay=0.22))
        errors = []
        def run():
            try:
                recorder.run()
            except Exception as exc:
                errors.append(exc)
        worker = threading.Thread(target=run)
        worker.start()
        timeout = time.monotonic() + 5
        while recorder.snapshot()["frames"] < 1 and time.monotonic() < timeout:
            time.sleep(0.01)
        time.sleep(0.25)
        recorder.stop()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        summary = json.loads(self.output.with_suffix(".summary.json").read_text())
        self.assertLess(abs(summary["video_seconds"] - summary["wall_clock_seconds"]), 0.101)

    def test_existing_output_is_never_overwritten(self):
        self.output.write_bytes(b"existing recording")
        recorder = Recorder(RecorderConfig(self.output, audio="none", duration=0.1), SyntheticSource)
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            recorder.run()
        self.assertEqual(self.output.read_bytes(), b"existing recording")

    def test_capture_failure_retains_recovery_files_and_no_final_output(self):
        recorder = Recorder(RecorderConfig(self.output, audio="none", duration=1),
                            lambda: SyntheticSource(fail_after=2))
        with self.assertRaisesRegex(RuntimeError, "Recovery files"):
            recorder.run()
        self.assertFalse(self.output.exists())
        self.assertEqual(recorder.snapshot()["status"], "Failed")
        self.assertTrue(list(self.root.glob(".recording-*/metrics.csv")))

    def test_mux_mixed_sample_rates_and_late_audio(self):
        video = self.root / "video.mp4"
        executable = locate_ffmpeg("ffmpeg")
        config = RecorderConfig(self.output, audio="none")
        writer = H264Writer(executable, video, 160, 90, config)
        for _ in range(10):
            writer.write(np.full((90, 160, 3), 20, dtype=np.uint8))
        writer.close()
        paths = []
        for index, rate in enumerate((8000, 48000)):
            path = self.root / f"track{index}.wav"
            pcm = (np.sin(2 * np.pi * 440 * np.arange(rate // 2) / rate) * 8000).astype("<i2")
            audio = TimelineWaveWriter(path, rate, 1)
            audio.append(pcm.tobytes(), 0.25)
            audio.close(1)
            paths.append(path)
        mux_recording(executable, video, paths, 1, self.output)
        info = self.probe(self.output)
        audio_stream = next(stream for stream in info["streams"] if stream["codec_type"] == "audio")
        self.assertEqual(audio_stream["codec_name"], "aac")
        self.assertAlmostEqual(float(audio_stream["duration"]), 1, places=2)
        decoded = subprocess.run([executable, "-v", "error", "-i", str(self.output), "-map", "0:a",
                                  "-ac", "1", "-ar", "48000", "-f", "s16le", "pipe:1"],
                                 check=True, capture_output=True, **subprocess_options()).stdout
        pcm = np.frombuffer(decoded, dtype="<i2")
        self.assertLess(np.abs(pcm[:10000].astype(float)).mean(), 10)
        self.assertGreater(np.abs(pcm[15000:30000].astype(float)).mean(), 2000)


if __name__ == "__main__":
    unittest.main()
