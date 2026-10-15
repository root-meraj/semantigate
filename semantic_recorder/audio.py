"""Windows WASAPI capture; callbacks enqueue PCM, a worker writes each WAV."""

from dataclasses import dataclass
import math
from pathlib import Path
import queue
import sys
import threading
import time
import wave


def load_backend():
    if sys.platform != "win32":
        raise RuntimeError("System/microphone capture currently supports Windows. Use --audio none here.")
    try:
        import pyaudiowpatch
    except ImportError as exc:
        raise RuntimeError("Audio capture needs PyAudioWPatch. Install requirements.txt or use --audio none.") from exc
    return pyaudiowpatch


def list_audio_devices() -> list[dict]:
    backend = load_backend()
    with backend.PyAudio() as audio:
        wasapi = audio.get_host_api_info_by_type(backend.paWASAPI)
        return [dict(device) for device in audio.get_device_info_generator()
                if device["hostApi"] == wasapi["index"] and device["maxInputChannels"] > 0]


class TimelineWaveWriter:
    """Place chunks on a common clock, filling gaps and trimming overlapping PCM.

    Ignore sub-20ms timestamp jitter after the first chunk. Larger discontinuities
    are repaired with silence/sample trimming and counted in recording metadata.
    """

    def __init__(self, path: Path, rate: int, channels: int):
        self.rate, self.channels = rate, channels
        self.sample_bytes = channels * 2
        self.frames = 0
        self.discontinuities = 0
        self._received = False
        self.wave = wave.open(str(path), "wb")
        self.wave.setnchannels(channels)
        self.wave.setsampwidth(2)
        self.wave.setframerate(rate)

    def _silence(self, frames: int) -> None:
        while frames > 0:
            count = min(frames, self.rate)
            self.wave.writeframesraw(bytes(count * self.sample_bytes))
            self.frames += count
            frames -= count

    def append(self, data: bytes, elapsed: float) -> None:
        if len(data) % self.sample_bytes:
            raise ValueError("Received an incomplete PCM audio frame.")
        target = round(elapsed * self.rate)
        if target < 0:
            skipped = min(-target, len(data) // self.sample_bytes)
            data = data[skipped * self.sample_bytes:]
            target += skipped
        if not data:
            return
        difference = target - self.frames
        if self._received and abs(difference) <= round(self.rate * 0.02):
            difference = 0
        if difference > 0:
            self._silence(difference)
            if self._received:
                self.discontinuities += 1
        elif difference < 0:
            data = data[min(-difference * self.sample_bytes, len(data)):]
            if self._received:
                self.discontinuities += 1
        self.wave.writeframesraw(data)
        self.frames += len(data) // self.sample_bytes
        self._received = True

    def close(self, duration: float) -> None:
        try:
            self._silence(max(0, math.ceil(duration * self.rate) - self.frames))
        finally:
            self.wave.close()


@dataclass
class AudioTrack:
    kind: str
    device: dict
    path: Path
    rate: int
    channels: int
    stream: object = None
    worker: threading.Thread | None = None
    error: str | None = None
    buffer_flags: int = 0
    timestamp_fallbacks: int = 0
    discontinuities: int = 0
    chunks: int = 0
    clock_offset: float = 0.0
    first_chunk_timing: dict | None = None
    input_latency_seconds: float = 0.0


class AudioCapture:
    def __init__(self, mode: str, directory: Path, system_device=None, microphone_device=None):
        self.tracks: list[AudioTrack] = []
        self._audio = None
        self._origin = 0.0
        self._duration = 0.0
        self._queues = []
        self._closed = False
        if mode == "none":
            return
        self._backend = load_backend()
        self._audio = self._backend.PyAudio()
        try:
            wasapi = self._audio.get_host_api_info_by_type(self._backend.paWASAPI)
            devices = []
            if mode in {"system", "both"}:
                device = (self._audio.get_default_wasapi_loopback() if system_device is None
                          else self._audio.get_device_info_by_index(system_device))
                if not device.get("isLoopbackDevice"):
                    raise RuntimeError("The selected system-audio device is not a WASAPI loopback input.")
                devices.append(("system", device))
            if mode in {"mic", "both"}:
                index = wasapi["defaultInputDevice"] if microphone_device is None else microphone_device
                device = self._audio.get_device_info_by_index(index)
                if device.get("isLoopbackDevice") or device["hostApi"] != wasapi["index"]:
                    raise RuntimeError("Choose a WASAPI microphone input from --list-devices.")
                devices.append(("mic", device))
            for kind, device in devices:
                channels = min(2, int(device["maxInputChannels"]))
                if channels < 1:
                    raise RuntimeError(f"The selected {kind} device has no audio input channels.")
                track = AudioTrack(kind, dict(device), directory / f"{kind}.wav",
                                   int(device["defaultSampleRate"]), channels)
                pending = queue.Queue(maxsize=256)
                self.tracks.append(track)
                self._queues.append(pending)
                track.stream = self._audio.open(
                    format=self._backend.paInt16, channels=channels, rate=track.rate,
                    input=True, input_device_index=int(device["index"]),
                    frames_per_buffer=1024, start=False,
                    stream_callback=self._callback(track, pending),
                )
                track.input_latency_seconds = track.stream.get_input_latency()
        except BaseException:
            self.close(0.0)
            raise

    def _callback(self, track, pending):
        def callback(data, count, timing, flags):
            try:
                # Anchor PortAudio's stream clock to the recorder before start.
                # Callback delivery can lag the callback's reported current
                # time, so delivery time is only a fallback for invalid ADCs.
                delivery = time.perf_counter() - self._origin
                current = timing.get("current_time", 0.0)
                adc = timing.get("input_buffer_adc_time", 0.0)
                offset = adc - current
                if current > 0 and adc > 0 and -2.0 <= offset <= 0.1:
                    elapsed = adc + track.clock_offset
                else:
                    elapsed = delivery - count / track.rate
                    track.timestamp_fallbacks += 1
                if track.first_chunk_timing is None:
                    track.first_chunk_timing = {"delivery_seconds": delivery, "sample_seconds": elapsed,
                                                "adc_minus_callback_seconds": offset,
                                                "reported_clock_delay_seconds": delivery - (current + track.clock_offset)}
                track.buffer_flags |= flags
                pending.put_nowait((data, elapsed))
                track.chunks += 1
                return None, self._backend.paContinue
            except Exception as exc:
                track.error = f"{track.kind} audio callback failed: {type(exc).__name__}: {exc}"
                return None, self._backend.paAbort
        return callback

    def _write_track(self, track, pending):
        writer = None
        try:
            writer = TimelineWaveWriter(track.path, track.rate, track.channels)
            while True:
                item = pending.get()
                if item is None:
                    break
                writer.append(*item)
        except Exception as exc:
            track.error = f"Could not save {track.kind} audio: {exc}"
        finally:
            if writer is not None:
                try:
                    writer.close(self._duration)
                    track.discontinuities = writer.discontinuities
                except Exception as exc:
                    track.error = f"Could not finalize {track.kind} audio: {exc}"

    def start(self, origin: float) -> None:
        self._origin = origin
        for track, pending in zip(self.tracks, self._queues):
            before = time.perf_counter()
            device_clock = track.stream.get_time()
            after = time.perf_counter()
            track.clock_offset = (before + after) / 2 - origin - device_clock
            track.worker = threading.Thread(target=self._write_track, args=(track, pending),
                                            name=f"audio-{track.kind}", daemon=True)
            track.worker.start()
            track.stream.start_stream()

    def check(self) -> None:
        for track in self.tracks:
            if track.error:
                raise RuntimeError(track.error)
            if not track.stream.is_active():
                raise RuntimeError(f"The {track.kind} audio stream stopped. Check its device connection.")

    def close(self, duration: float) -> None:
        if self._closed:
            return
        self._closed = True
        self._duration = duration
        for track, pending in zip(self.tracks, self._queues):
            if track.stream is not None:
                try:
                    if track.stream.is_active():
                        track.stream.stop_stream()
                except Exception as exc:
                    track.error = f"Could not stop {track.kind} audio: {exc}"
                finally:
                    try:
                        track.stream.close()
                    except Exception as exc:
                        track.error = f"Could not close {track.kind} audio: {exc}"
            if track.worker is not None:
                try:
                    pending.put(None, timeout=2)
                except queue.Full:
                    track.error = f"The {track.kind} audio writer stopped responding."
                track.worker.join(timeout=10)
                if track.worker.is_alive():
                    track.error = f"The {track.kind} audio writer did not finish."
        if self._audio is not None:
            try:
                self._audio.terminate()
            except Exception as exc:
                for track in self.tracks:
                    track.error = track.error or f"Could not release the audio backend: {exc}"

    def metadata(self) -> list[dict]:
        return [{"kind": track.kind, "device": track.device["name"], "rate": track.rate,
                 "channels": track.channels, "chunks": track.chunks,
                 "buffer_status_flags": track.buffer_flags,
                 "timestamp_fallbacks": track.timestamp_fallbacks,
                 "input_latency_seconds": track.input_latency_seconds,
                 "first_chunk_timing": track.first_chunk_timing,
                 "discontinuities": track.discontinuities, "error": track.error}
                for track in self.tracks]
