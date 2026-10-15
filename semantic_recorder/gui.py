from datetime import datetime
import os
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .audio import list_audio_devices
from .config import RecorderConfig
from .engine import Recorder, list_monitors


AUDIO_MODES = {"System audio": "system", "Microphone": "mic", "System + microphone": "both", "Silent": "none"}


def default_output() -> Path:
    return Path.cwd() / "recordings" / f"recording_{datetime.now():%Y%m%d_%H%M%S_%f}.mp4"


class RecorderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.recorder = self.worker = None
        self.closing = False
        self._photo = self._last_status = None
        self._editable = []
        self._device_map = {"Automatic": None}
        self._monitor_map = {"Primary monitor": None}
        self.output = tk.StringVar(value=str(default_output()))
        self.audio = tk.StringVar(value="System audio")
        self.monitor = tk.StringVar(value="Primary monitor")
        self.system_device = tk.StringVar(value="Automatic")
        self.mic_device = tk.StringVar(value="Automatic")
        self.fps = tk.StringVar(value="10")
        self.crf = tk.StringVar(value="23")
        self.minimize = tk.BooleanVar(value=True)
        self.show_preview = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Ready to record")
        self.detail = tk.StringVar(value="Static frames are held; audio continues throughout.")
        self.device_note = tk.StringVar()
        root.title("Semantic Video Recorder")
        root.geometry("860x740")
        root.minsize(760, 650)
        root.configure(bg="#111827")
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#111827")
        style.configure("TLabel", background="#111827", foreground="#e5e7eb", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"), foreground="#f9fafb")
        style.configure("Status.TLabel", font=("Segoe UI", 15, "bold"), foreground="#5eead4")
        style.configure("Muted.TLabel", foreground="#9ca3af", font=("Segoe UI", 9))
        style.configure("TCheckbutton", background="#111827", foreground="#e5e7eb")
        style.map("TCheckbutton", background=[("active", "#1f2937")])
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8))
        container = ttk.Frame(root, padding=24)
        container.pack(fill="both", expand=True)
        container.columnconfigure(1, weight=1)
        ttk.Label(container, text="Semantic Video Recorder", style="Title.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(container, text="Motion-gated screen capture · H.264 + AAC", style="Muted.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(4, 20))
        self._field(container, 2, "Save recording", ttk.Entry(container, textvariable=self.output))
        self.browse_button = ttk.Button(container, text="Browse…", command=self.browse)
        self.browse_button.grid(row=2, column=2, padx=(8, 0))
        self._editable.append(self.browse_button)
        self.monitor_combo = ttk.Combobox(container, textvariable=self.monitor, state="readonly",
                                         values=["Primary monitor"])
        self._field(container, 3, "Screen", self.monitor_combo)
        self._field(container, 4, "Audio", ttk.Combobox(container, textvariable=self.audio,
                    state="readonly", values=list(AUDIO_MODES)))
        self.system_combo = ttk.Combobox(container, textvariable=self.system_device,
                                        state="readonly", values=["Automatic"])
        self._field(container, 5, "System audio device", self.system_combo)
        self.mic_combo = ttk.Combobox(container, textvariable=self.mic_device,
                                     state="readonly", values=["Automatic"])
        self._field(container, 6, "Microphone device", self.mic_combo)
        self._field(container, 7, "Frames per second", ttk.Spinbox(container, from_=1, to=60,
                    textvariable=self.fps, width=8))
        self._field(container, 8, "Quality (CRF)", ttk.Spinbox(container, from_=0, to=51,
                    textvariable=self.crf, width=8))
        ttk.Label(container, text="CRF 18–23 suits slide recordings. Lower values keep more detail.",
                  style="Muted.TLabel").grid(row=9, column=1, columnspan=2, sticky="w")
        ttk.Label(container, textvariable=self.device_note, style="Muted.TLabel", wraplength=720).grid(
            row=10, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.minimize_check = ttk.Checkbutton(container, text="Minimize while recording to keep controls out of the capture",
                                              variable=self.minimize)
        self.minimize_check.grid(row=11, column=0, columnspan=3, sticky="w", pady=(12, 4))
        self._editable.append(self.minimize_check)
        ttk.Checkbutton(container, text="Show recorded-frame preview", variable=self.show_preview).grid(
            row=12, column=0, columnspan=3, sticky="w")
        controls = ttk.Frame(container)
        controls.grid(row=13, column=0, columnspan=3, sticky="w", pady=18)
        self.start_button = ttk.Button(controls, text="●  Start recording", command=self.start)
        self.start_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(controls, text="■  Stop & save", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 8))
        self.folder_button = ttk.Button(controls, text="Open folder", command=self.open_folder)
        self.folder_button.pack(side="left", padx=(0, 8))
        self.refresh_button = ttk.Button(controls, text="Refresh devices", command=self.refresh_devices)
        self.refresh_button.pack(side="left")
        self._editable.append(self.refresh_button)
        ttk.Label(container, textvariable=self.status, style="Status.TLabel").grid(
            row=14, column=0, columnspan=3, sticky="w")
        ttk.Label(container, textvariable=self.detail, wraplength=780).grid(
            row=15, column=0, columnspan=3, sticky="w", pady=(6, 8))
        self.preview_label = ttk.Label(container)
        self.preview_label.grid(row=16, column=0, columnspan=3)
        ttk.Label(container, text="Md Meraj Ansari · Symbiosis Institute of Technology, Hyderabad",
                  style="Muted.TLabel").grid(row=17, column=0, columnspan=3, sticky="w", pady=(12, 0))
        self.refresh_devices()
        root.after(200, self.poll)

    def _field(self, container, row, text, widget):
        ttk.Label(container, text=text).grid(row=row, column=0, sticky="w", padx=(0, 16), pady=6)
        widget.grid(row=row, column=1, sticky="ew", pady=6)
        self._editable.append(widget)

    def refresh_devices(self):
        notes = []
        try:
            self._monitor_map = {"Primary monitor": None}
            for monitor in list_monitors():
                label = f"Monitor {monitor['index']} · {monitor['width']} × {monitor['height']}"
                self._monitor_map[label] = monitor["index"]
            self.monitor_combo.configure(values=list(self._monitor_map))
            if self.monitor.get() not in self._monitor_map:
                self.monitor.set("Primary monitor")
        except Exception as exc:
            notes.append(f"Screens: {exc}")
        try:
            self._device_map = {"Automatic": None}
            system = ["Automatic"]
            microphones = ["Automatic"]
            for device in list_audio_devices():
                label = f"{device['index']}: {device['name']}"
                self._device_map[label] = int(device["index"])
                (system if device.get("isLoopbackDevice") else microphones).append(label)
            self.system_combo.configure(values=system)
            self.mic_combo.configure(values=microphones)
            if self.system_device.get() not in system:
                self.system_device.set("Automatic")
            if self.mic_device.get() not in microphones:
                self.mic_device.set("Automatic")
        except Exception as exc:
            notes.append(f"Audio: {exc}")
        self.device_note.set(" · ".join(notes) or "Automatic uses the Windows default speaker and microphone.")

    def browse(self):
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".mp4",
                filetypes=[("MP4 video", "*.mp4")], initialfile=Path(self.output.get()).name)
        if path:
            self.output.set(path)

    def _set_busy(self, busy):
        for widget in self._editable:
            widget.configure(state="disabled" if busy else ("readonly" if isinstance(widget, ttk.Combobox) else "normal"))
        self.start_button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")

    def start(self):
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            config = RecorderConfig(output=Path(self.output.get()), fps=float(self.fps.get()),
                crf=int(self.crf.get()), monitor=self._monitor_map[self.monitor.get()],
                audio=AUDIO_MODES[self.audio.get()], system_device=self._device_map[self.system_device.get()],
                microphone_device=self._device_map[self.mic_device.get()])
            config.validate()
            if config.output.exists():
                raise ValueError("That recording already exists. Choose a new filename.")
        except (ValueError, KeyError) as exc:
            messagebox.showerror("Check recording settings", str(exc), parent=self.root)
            return
        self.recorder = Recorder(config)
        self._last_status = None
        self._set_busy(True)
        self.status.set("Preparing recording…")
        self.worker = threading.Thread(target=self._record, name="screen-recorder", daemon=True)
        self.worker.start()
        if self.minimize.get():
            self.root.iconify()

    def _record(self):
        try:
            self.recorder.run()
        except Exception:
            pass  # The engine exposes the error; only Tk's main thread displays it.

    def stop(self):
        if self.recorder is not None:
            self.recorder.stop()
            self.stop_button.configure(state="disabled")
            self.status.set("Stopping and saving…")

    def poll(self):
        if self.recorder is not None:
            stats = self.recorder.snapshot()
            current = stats["status"]
            busy = self.worker is not None and self.worker.is_alive()
            if current in {"ACTIVE", "IDLE", "REFRESH"}:
                elapsed = int(stats["elapsed"])
                self.status.set(f"{current}  ·  {elapsed // 60:02d}:{elapsed % 60:02d}")
                reused = 100 * stats["repeated"] / max(1, stats["frames"])
                self.detail.set(f"{stats['frames']} frames · {reused:.0f}% reused · {stats['late_frames']} late timeline slots")
            elif current == "Finalizing":
                self.status.set("Saving video and audio…")
            if not busy and current in {"Saved", "Failed"} and self._last_status != current:
                self._set_busy(False)
                self.root.deiconify()
                if current == "Saved":
                    self.status.set("Recording saved")
                    warning = " · ".join(stats["warnings"])
                    self.detail.set(stats["output"] + (f"\n{warning}" if warning else ""))
                    self.output.set(str(default_output()))
                else:
                    self.status.set("Recording failed")
                    self.detail.set(stats["error"])
                    if not self.closing:
                        messagebox.showerror("Recording failed", stats["error"], parent=self.root)
            # A terminal status can appear just before the worker exits. Keep
            # it pending until the completion UI has actually been displayed.
            if current not in {"Saved", "Failed"} or not busy:
                self._last_status = current
            if self.show_preview.get() and self.root.state() != "iconic":
                preview = self.recorder.preview()
                if preview is not None:
                    image = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
                    image.thumbnail((400, 225))
                    self._photo = ImageTk.PhotoImage(image)
                    self.preview_label.configure(image=self._photo)
            else:
                self.preview_label.configure(image="")
            if self.closing and not busy:
                self.root.destroy()
                return
        self.root.after(200, self.poll)

    def open_folder(self):
        saved = self.recorder.snapshot()["output"] if self.recorder is not None else None
        folder = Path(saved or self.output.get()).expanduser().resolve().parent
        folder.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(str(folder))

    def close(self):
        if self.worker is not None and self.worker.is_alive():
            self.closing = True
            self.stop()
            self.detail.set("Finishing the recording before closing this window…")
        else:
            self.root.destroy()


def launch():
    root = tk.Tk()
    RecorderApp(root)
    root.mainloop()
