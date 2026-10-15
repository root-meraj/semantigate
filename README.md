# Semantic Video Recorder & Motion-Gated Inference

**Lead developer:** Md Meraj Ansari  
**Institution:** Symbiosis Institute of Technology, Hyderabad

The Phase 3 desktop recorder captures a monitor, holds redundant frames during
static periods, and writes an H.264 MP4 with optional system audio and microphone
audio. The original webcam motion/gesture experiments remain in this folder.

## Run on Windows

Requires Python 3.10+ with Tkinter and FFmpeg with the `libx264` encoder on PATH.
FFmpeg is already available on this workstation. Use the project environment:

```powershell
cd B:\motion_gate
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe smart_compressor.py
```

Double-click **run_recorder.bat** to launch the desktop interface using the
existing environment. Choose the monitor, destination, audio mode, FPS, and
quality, then press **Start recording**. The window minimizes by default to
keep its changing controls and preview out of the captured monitor. Restore it
from the taskbar and press **Stop & save**. Closing the window while recording
also stops and saves before exiting.

System audio is the default. Select **System + microphone** to include both
sides of a meeting, **Microphone** for narration, or **Silent** for video only.
Automatic devices follow the Windows default speakers and WASAPI microphone at
recording start. Refresh/select devices after changing headphones or speakers.

## Record from the terminal

```powershell
# List screen indices and eligible audio inputs.
.\venv\Scripts\python.exe smart_compressor.py --list-devices

# Record both audio sources; Ctrl+C stops and saves.
.\venv\Scripts\python.exe smart_compressor.py --cli --audio both

# A timed, silent recording, with explicit quality and destination.
.\venv\Scripts\python.exe smart_compressor.py --cli --audio none --duration 60 --fps 10 --crf 23 --output recordings\demo.mp4

# Select devices by indices printed by --list-devices.
.\venv\Scripts\python.exe smart_compressor.py --cli --audio both --system-device 10 --microphone-device 9
```

Indices are machine-specific. Run `--help` for monitor, encoder, sensitivity,
refresh, and executable options. The package also launches with
`python -m semantic_recorder`. Supplying recording options automatically selects
terminal recording, even without `--cli`.

## How timing and gating work

- Capture uses a monotonic clock and defaults to 10 FPS. Every elapsed frame
  slot is encoded. Slow capture fills missing slots with the previously displayed
  frame; capture speed no longer sets the playback duration. Video length is
  quantized to FPS, normally within one frame of the Stop time.
- The pixel gate downsizes analysis to 640 × 360, uses an 11 × 11 Gaussian blur,
  thresholds intensity changes at 15, dilates the mask, and updates the picture
  when more than 1,000 pixels change. It compares with the displayed frame so
  a newly static slide does not keep triggering until a background model settles.
- A refresh every two seconds makes small edits visible even when they fall
  below the gate threshold. Tune `--changed-pixels`, `--pixel-threshold`, or
  `--refresh-seconds` for the material being recorded. Tiny animations can still
  be suppressed between refreshes.
- FFmpeg encodes every timeline slot using H.264 (`libx264`, `veryfast`, CRF 23,
  `yuv420p`). Reusing identical pictures can aid inter-frame compression; the
  encoder still runs, so frame reuse alone does not prove CPU or power savings.
  Odd monitor dimensions are padded by at most one row/column for compatibility.
- WASAPI audio callbacks enqueue 16-bit PCM; separate workers write WAV tracks.
  PortAudio device clocks are mapped to the same monotonic timeline, preserving
  startup offsets. Missing audio intervals receive silence; overlaps are trimmed.
  FFmpeg mixes both sources with normalization, pads/trims them to the video
  duration, and encodes AAC. Video is copied during this final assembly.

Final outputs use `faststart` for convenient playback. This implementation keeps
constant frame rate for predictable playback and audio assembly. It does not yet
use a variable frame rate container to skip encoded frame slots.

## Files, telemetry, and recovery

Each successful recording produces:

| File | Contents |
| --- | --- |
| `recording_….mp4` | H.264 video, plus AAC when audio is selected |
| `recording_….metrics.csv` | Capture state, changed pixels, frame counts, late slots, recorder-process CPU/RSS, and system RAM |
| `recording_….summary.json` | Settings, wall-clock/video durations, frame reuse, file bytes, selected devices, and audio diagnostics |

The app refuses to overwrite existing video or sidecar files. Default names
include a timestamp and microseconds. Encoding happens in a temporary directory
beside the output. Failed runs preserve that directory, partial video, WAVs,
metrics, and FFmpeg logs; the error displays its path. Successful runs clean up
temporary files after publishing completed outputs.

PCM intermediates need approximately **0.69 GB per hour per stereo 48 kHz source**,
in addition to the MP4. Final assembly currently has a 60-second timeout, so very
long recordings/slow storage may require that limit to be raised. WAV intermediates
also inherit the usual RIFF size limit. Long-session soak testing and streaming
audio muxing remain future work.

Audio capture currently targets Windows. Silent recording may work elsewhere
if screen capture and FFmpeg are available. Driver/hardware latency can leave
an audible offset even with aligned file durations. Use the check below on the
actual recording device; Bluetooth devices especially need their own validation.
No incoming PCM during a silent speaker interval is expected and produces silence.
Buffer/timestamp diagnostics and repaired discontinuities appear in the summary.

## Verification

```powershell
# Synthetic gate, timing, audio alignment/mixing, GUI state, and failure tests.
.\venv\Scripts\python.exe -m unittest discover -s tests -v

# Plays a quiet 0.6-second tone; records synthetic video and real speaker loopback.
.\venv\Scripts\python.exe -m tools.check_loopback
```

Local verification on **12 October 2026**:

- The regression suite checks static frames, slide transitions, small edits,
  noisy frames, fast/slow capture, Stop timing, odd dimensions, mixed audio
  sample rates, delayed audio, existing-file protection, recovery, and GUI completion.
- A real 1920 × 1080 desktop run produced 30 H.264 frames and a 3.00-second AAC
  track for a 3.00-second recording with system + microphone capture enabled.
- A clocked loopback tone check captured audible PCM with an observed picture/
  audio onset difference around 0.13 seconds on the built-in Realtek device.
  The check reports the measured difference and uses a 250 ms tolerance; it is
  not a guarantee for other devices or long recordings.

Frame reuse percentage is **not** storage savings. Compare the same input with
identical FPS/CRF/preset using `--no-gate` to measure the incremental gate benefit
over normal H.264. Measure wall power or device energy for a power claim. Recorder
process CPU in the CSV excludes the separate FFmpeg encoder process.

The historical `research_metrics.csv` currently contains 79 ACTIVE samples over
3.83 seconds and no IDLE samples. It cannot establish an IDLE-versus-ACTIVE benefit.

## Project layout and next milestone

`semantic_recorder/` contains configuration, the gate, audio capture, FFmpeg
encoding, the recorder engine, CLI, and Tkinter interface. `smart_compressor.py`
is the entry point. `smart_compressor_mvp.py` preserves the original MP4V MVP.
`motion_gate.py`, `motion_gate_ai.py`, and `python gesture_pipeline.py` retain the
Phase 1 camera/MediaPipe experiments; recorder requirements do not reinstall
their MediaPipe dependencies.

The YOLO data workflow is now available in `semantic_dataset/` and `training/`:
screen/video collection, a local box annotator, validation, session-isolated
splits, ZIP export, and a Colab training notebook. Start with
[training/README.md](training/README.md) and the
[annotation guide](training/ANNOTATION_GUIDE.md). Launch the annotator with
`run_annotator.bat` after collecting candidates.

The agent reviewed all 148 real pilot screenshots and saved corrected labels in
`datasets/screen_elements/source_reviewed_v2/`: 126 included images and 22 excluded
transitions. Manual annotation of these captures is complete. Only one call has
confirmed active non-slide sharing, so two additional independent practice calls
with all three content types are needed for the current training split. See the
[completed review](datasets/screen_elements/annotation_review/full_review/RESULT.md).
No custom model has been trained yet. After the remaining data gap is filled,
train using `training/train_screen_elements_colab.ipynb`. The recorder keeps
the pixel gate as a baseline. The next engineering milestone is an evaluated
region retention policy using the trained checkpoint, with measurements of
readability, missed transitions, inference overhead, and storage.

Implementation references: [PyAudioWPatch loopback API](https://github.com/s0d3s/PyAudioWPatch),
[PortAudio callback timestamps](https://portaudio.com/docs/v19-doxydocs/structPaStreamCallbackTimeInfo.html),
[FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html), and
[FFmpeg H.264 encoder options](https://ffmpeg.org/ffmpeg-codecs.html).
