from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from . import CLASSES
from .core import Box, exclude_sample, image_info, load_samples, read_labels, save_labels


COLORS = ("#38bdf8", "#fbbf24", "#a78bfa")


class Annotator:
    def __init__(self, root: tk.Tk, source: Path):
        self.root = root
        self.samples = load_samples(source)
        self.index = next((i for i, sample in enumerate(self.samples)
                           if not sample.excluded and not sample.label.exists()), 0)
        self.boxes = []
        self.image = self.photo = None
        self.drag_start = self.drag_id = None
        self.scale = 1.0
        self.offset = (0, 0)
        self.dirty = False
        self.status = tk.StringVar()
        self.progress = tk.StringVar()
        self.guidance = tk.StringVar()
        self.class_name = tk.StringVar(value=f"0: {CLASSES[0]}")
        root.title("Screen Element Annotator")
        root.geometry("1280x820")
        root.minsize(900, 600)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.protocol("WM_DELETE_WINDOW", self.close)
        top = ttk.Frame(root, padding=12)
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(top, textvariable=self.status, font=("Segoe UI", 13, "bold")).pack(anchor="w")
        ttk.Label(top, textvariable=self.progress).pack(anchor="w", pady=(4, 0))
        ttk.Label(top, textvariable=self.guidance, wraplength=1100, foreground="#9a3412").pack(anchor="w", pady=(4, 0))
        self.canvas = tk.Canvas(root, background="#111827", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=(12, 0), pady=(0, 12))
        side = ttk.Frame(root, padding=12)
        side.grid(row=1, column=1, sticky="ns")
        ttk.Label(side, text="Choose class, then drag a box", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.class_combo = ttk.Combobox(side, textvariable=self.class_name, state="readonly", width=29,
                                       values=[f"{i}: {name}" for i, name in enumerate(CLASSES)])
        self.class_combo.pack(fill="x", pady=10)
        ttk.Label(side, text="0  Camera video feed and its background\n1  Slide canvas\n2  Shared app/desktop (non-slide)\n\nAdd separate boxes when a camera\nand shared content are both visible.",
                  justify="left", wraplength=250).pack(anchor="w")
        ttk.Label(side, text="Boxes in this image", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(18, 4))
        self.box_list = tk.Listbox(side, height=8, width=32, exportselection=False)
        self.box_list.pack(fill="x")
        ttk.Button(side, text="Delete selected box", command=self.delete_selected).pack(fill="x", pady=4)
        ttk.Button(side, text="Undo last box  [U]", command=self.undo).pack(fill="x", pady=4)
        ttk.Separator(side).pack(fill="x", pady=12)
        ttk.Button(side, text="Save & next  [S]", command=self.save_next).pack(fill="x", pady=4)
        ttk.Button(side, text="Mark negative & next  [N]", command=self.negative_next).pack(fill="x", pady=4)
        self.exclude_button = ttk.Button(side, text="Exclude from dataset", command=self.toggle_excluded)
        self.exclude_button.pack(fill="x", pady=4)
        navigation = ttk.Frame(side)
        navigation.pack(fill="x", pady=12)
        ttk.Button(navigation, text="← Previous", command=lambda: self.navigate(-1)).pack(side="left")
        ttk.Button(navigation, text="Next →", command=lambda: self.navigate(1)).pack(side="right")
        ttk.Label(side, text="Class keys: 1, 2, 3\nNavigation: left/right arrows\n\nMissing labels are unreviewed.\nUse Mark negative only when none\nof the three classes is visible.\n\nSee training/ANNOTATION_GUIDE.md\nfor class and overlap rules.",
                  justify="left", wraplength=250).pack(anchor="w", pady=(12, 0))
        self.canvas.bind("<Configure>", lambda _: self.draw())
        self.canvas.bind("<ButtonPress-1>", self.begin_box)
        self.canvas.bind("<B1-Motion>", self.drag_box)
        self.canvas.bind("<ButtonRelease-1>", self.finish_box)
        root.bind("<Left>", lambda _: self.navigate(-1))
        root.bind("<Right>", lambda _: self.navigate(1))
        root.bind("s", lambda _: self.save_next())
        root.bind("n", lambda _: self.negative_next())
        root.bind("u", lambda _: self.undo())
        for key in (1, 2, 3):
            root.bind(str(key), lambda _, index=key - 1: self.class_combo.current(index))
        self.load_current()

    @property
    def sample(self):
        return self.samples[self.index]

    def load_current(self):
        with Image.open(self.sample.image) as source:
            self.image = source.convert("RGB")
        self.boxes = []
        if self.sample.label.exists():
            try:
                self.boxes = read_labels(self.sample.label)
            except ValueError as exc:
                messagebox.showerror("Label needs repair", str(exc), parent=self.root)
        self.dirty = False
        self.drag_start = self.drag_id = None
        self.update_status()
        self.draw()

    def update_status(self):
        state = "EXCLUDED" if self.sample.excluded else "reviewed" if self.sample.label.exists() else "unreviewed"
        self.status.set(f"{self.index + 1}/{len(self.samples)} · {self.sample.session} · {self.sample.image.name} · {state}")
        included = [sample for sample in self.samples if not sample.excluded]
        reviewed = sum(sample.label.exists() for sample in included)
        self.progress.set(f"{reviewed}/{len(included)} included images reviewed · {len(self.samples) - len(included)} excluded")
        if any(box.class_id == 0 and box.width * box.height > 0.8 for box in self.boxes):
            self.guidance.set("Camera box covers most of the screenshot. Check its bounds: include the video feed, leaving browser and meeting controls outside.")
        else:
            self.guidance.set("Draw one box for each visible region. Camera-off/error placeholders are not Speaker_Video.")
        self.exclude_button.configure(text="Include in dataset" if self.sample.excluded else "Exclude from dataset")
        self.box_list.delete(0, "end")
        for index, box in enumerate(self.boxes):
            self.box_list.insert("end", f"{index + 1}. {CLASSES[box.class_id]}")

    def draw(self):
        if self.image is None:
            return
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.scale = min(width / self.image.width, height / self.image.height)
        shown = (max(1, round(self.image.width * self.scale)), max(1, round(self.image.height * self.scale)))
        self.offset = ((width - shown[0]) / 2, (height - shown[1]) / 2)
        self.photo = ImageTk.PhotoImage(self.image.resize(shown, Image.Resampling.LANCZOS))
        self.canvas.delete("all")
        self.canvas.create_image(*self.offset, image=self.photo, anchor="nw")
        for box in self.boxes:
            x0 = self.offset[0] + (box.x - box.width / 2) * self.image.width * self.scale
            y0 = self.offset[1] + (box.y - box.height / 2) * self.image.height * self.scale
            x1 = x0 + box.width * self.image.width * self.scale
            y1 = y0 + box.height * self.image.height * self.scale
            color = COLORS[box.class_id]
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2)
            self.canvas.create_text(x0 + 4, y0 + 4, text=CLASSES[box.class_id], fill=color, anchor="nw",
                                    font=("Segoe UI", 10, "bold"))

    def image_point(self, x, y):
        return (max(0, min(self.image.width, (x - self.offset[0]) / self.scale)),
                max(0, min(self.image.height, (y - self.offset[1]) / self.scale)))

    def begin_box(self, event):
        left, top = self.offset
        if not (left <= event.x <= left + self.image.width * self.scale
                and top <= event.y <= top + self.image.height * self.scale):
            return
        self.canvas.focus_set()
        self.drag_start = self.image_point(event.x, event.y)
        self.drag_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y,
                    outline=COLORS[self.class_combo.current()], width=2, dash=(4, 2))

    def drag_box(self, event):
        if self.drag_start is not None and self.drag_id is not None:
            x, y = self.image_point(event.x, event.y)
            x0, y0 = self.drag_start
            self.canvas.coords(self.drag_id, self.offset[0] + x0 * self.scale, self.offset[1] + y0 * self.scale,
                               self.offset[0] + x * self.scale, self.offset[1] + y * self.scale)

    def finish_box(self, event):
        if self.drag_start is None:
            return
        x, y = self.image_point(event.x, event.y)
        x0, y0 = self.drag_start
        self.drag_start = self.drag_id = None
        left, right = sorted((x0, x))
        top, bottom = sorted((y0, y))
        if right - left >= 3 and bottom - top >= 3:
            self.boxes.append(Box(self.class_combo.current(), (left + right) / 2 / self.image.width,
                (top + bottom) / 2 / self.image.height, (right - left) / self.image.width,
                (bottom - top) / self.image.height))
            self.dirty = True
        self.update_status()
        self.draw()

    def delete_selected(self):
        selected = self.box_list.curselection()
        if selected:
            self.boxes.pop(selected[0])
            self.dirty = True
            self.update_status()
            self.draw()

    def undo(self):
        if self.boxes:
            self.boxes.pop()
            self.dirty = True
            self.update_status()
            self.draw()

    def save(self, negative=False):
        try:
            if self.sample.excluded:
                raise ValueError("Include this image before saving it as a training example.")
            info = image_info(self.sample.image)
            expected = self.sample.metadata
            if info != (expected["width"], expected["height"], expected["pixel_sha256"]):
                raise ValueError("The image changed after collection. Import edited files as a new session.")
            if negative and self.boxes:
                raise ValueError("Remove the boxes before marking this image as negative.")
            save_labels(self.sample.label, self.boxes, negative=negative)
            self.dirty = False
            self.update_status()
            return True
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot save annotation", str(exc), parent=self.root)
            return False

    def save_next(self):
        if self.save():
            self.navigate(1)

    def negative_next(self):
        if self.save(negative=True):
            self.navigate(1)

    def confirm_navigation(self):
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel("Unsaved boxes", "Save these boxes before leaving this image?", parent=self.root)
        return self.save() if answer else answer is not None

    def navigate(self, step):
        if self.confirm_navigation():
            self.index = max(0, min(len(self.samples) - 1, self.index + step))
            self.load_current()

    def toggle_excluded(self):
        if not self.confirm_navigation():
            return
        exclude_sample(self.sample, not self.sample.excluded)
        self.load_current()

    def close(self):
        if self.confirm_navigation():
            self.root.destroy()


def launch(source: Path):
    # Fail before opening a blank window if collection has not been run yet.
    load_samples(source)
    root = tk.Tk()
    Annotator(root, source)
    root.mainloop()
