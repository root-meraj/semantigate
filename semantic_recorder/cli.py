import argparse
from datetime import datetime
from pathlib import Path
import signal
import sys

from .config import RecorderConfig
from .engine import Recorder, list_monitors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Motion-gated H.264 screen recorder. Opens the GUI by default.")
    parser.add_argument("--cli", action="store_true", help="Record without the desktop interface; Ctrl+C stops and saves")
    parser.add_argument("--list-devices", action="store_true", help="List screens and Windows WASAPI audio inputs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audio", choices=["none", "system", "mic", "both"], default="system")
    parser.add_argument("--system-device", type=int, help="WASAPI loopback index from --list-devices")
    parser.add_argument("--microphone-device", type=int, help="WASAPI microphone index from --list-devices")
    parser.add_argument("--monitor", type=int, help="Monitor index (default: primary)")
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"], default="veryfast")
    parser.add_argument("--duration", type=float, help="Stop automatically after this many seconds")
    parser.add_argument("--changed-pixels", type=int, default=1000)
    parser.add_argument("--pixel-threshold", type=int, default=15)
    parser.add_argument("--refresh-seconds", type=float, default=2)
    parser.add_argument("--no-gate", action="store_true", help="Encode every capture, for controlled comparisons")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable name or full path")
    args = parser.parse_args(argv)
    if args.list_devices:
        try:
            for monitor in list_monitors():
                print(f"Screen {monitor['index']}: {monitor['width']}x{monitor['height']} "
                      f"at ({monitor['left']}, {monitor['top']})")
            from .audio import list_audio_devices
            for device in list_audio_devices():
                kind = "system loopback" if device.get("isLoopbackDevice") else "microphone"
                print(f"Audio {device['index']}: {device['name']} ({kind})")
            return 0
        except Exception as exc:
            print(f"Device listing: {exc}", file=sys.stderr)
            return 1
    recording_options = {"--output", "--audio", "--system-device", "--microphone-device", "--monitor",
                         "--fps", "--crf", "--preset", "--duration", "--changed-pixels", "--pixel-threshold",
                         "--refresh-seconds", "--no-gate", "--ffmpeg"}
    supplied = sys.argv[1:] if argv is None else argv
    if not args.cli and not any(value.split("=", 1)[0] in recording_options for value in supplied):
        from .gui import launch
        launch()
        return 0
    output = args.output or Path("recordings") / f"recording_{datetime.now():%Y%m%d_%H%M%S_%f}.mp4"
    config = RecorderConfig(output=output, fps=args.fps, crf=args.crf, preset=args.preset,
        monitor=args.monitor, audio=args.audio, system_device=args.system_device,
        microphone_device=args.microphone_device, duration=args.duration,
        changed_pixels=args.changed_pixels, pixel_threshold=args.pixel_threshold,
        refresh_seconds=args.refresh_seconds, gate_enabled=not args.no_gate, ffmpeg=args.ffmpeg)
    recorder = Recorder(config)
    previous = signal.signal(signal.SIGINT, lambda *_: recorder.stop())
    try:
        print(f"Recording to {output.resolve()} | {config.fps:g} FPS | audio={config.audio}", flush=True)
        print("Press Ctrl+C once to stop and save.", flush=True)
        summary = recorder.run()
        print(f"Saved {output.resolve()} ({summary['video_seconds']:.2f}s, "
              f"{summary['output_bytes'] / 1024 ** 2:.2f} MiB, {summary['frame_reuse_percent']:.1f}% frames reused)")
        for warning in recorder.snapshot()["warnings"]:
            print(f"Warning: {warning}")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous)
