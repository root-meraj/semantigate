import ast
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

from semantic_dataset.core import (Box, SessionWriter, exclude_sample, export_bundle, inspect_source,
    load_samples, prepare_dataset, read_labels, save_labels, validate_prepared, write_json)
from semantic_dataset.collect import CandidateFilter, capture_screen, extract_video, import_images, validate_collection
from training.train_yolo import train, training_settings


def seed_source(root: Path, groups=3, per_group=4, reviewed=True):
    for group in range(groups):
        writer = SessionWriter(root, f"session_{group}", {"kind": "test-fixture"})
        for index in range(per_group):
            image = Image.fromarray(np.random.default_rng(group * 100 + index).integers(
                0, 256, size=(60, 100, 3), dtype=np.uint8))
            path = writer.add(image, float(index))
            if reviewed:
                save_labels(writer.directory / "labels" / path.with_suffix(".txt").name,
                    [Box(0, .15, .5, .2, .5), Box(1, .5, .5, .3, .5), Box(2, .85, .5, .2, .5)])


class LabelTests(unittest.TestCase):
    def test_invalid_boxes(self):
        for box in (Box(3, .5, .5, .2, .2), Box(0, .1, .5, .5, .2), Box(0, .5, .5, 0, .2),
                    Box(0, float("nan"), .5, .2, .2), Box(0, .5, .5, .2, float("inf"))):
            with self.subTest(box=box), self.assertRaises(ValueError):
                box.validate()

    def test_missing_label_differs_from_explicit_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            with self.assertRaisesRegex(ValueError, "Unreviewed"):
                read_labels(path)
            with self.assertRaisesRegex(ValueError, "Mark negative"):
                save_labels(path, [])
            save_labels(path, [], negative=True)
            self.assertEqual(read_labels(path), [])

    def test_strict_class_and_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            for text in ("1.0 .5 .5 .2 .2", "0 .5 .5 .2 .2\n0 .5 .5 .2 .2", "0 .5 .5 .2"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_labels(path)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source = self.root / "source"
        self.output = self.root / "prepared"

    def tearDown(self):
        self.directory.cleanup()

    def test_collect_deduplicates_decoded_pixels(self):
        writer = SessionWriter(self.source, "zoom_01", {})
        image = Image.new("RGB", (40, 20), "white")
        self.assertIsNotNone(writer.add(image))
        self.assertIsNone(writer.add(image.copy()))
        samples = load_samples(self.source)
        self.assertEqual(len(samples), 1)
        self.assertFalse(samples[0].label.exists())

    def test_session_cannot_overwrite_or_escape_root(self):
        SessionWriter(self.source, "zoom", {})
        with self.assertRaises(FileExistsError):
            SessionWriter(self.source, "zoom", {})
        for name in ("../outside", "CON", "meet/01", "", "x:y"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                SessionWriter(self.source, name, {})

    def test_unreviewed_data_cannot_prepare(self):
        seed_source(self.source, reviewed=False)
        with self.assertRaisesRegex(ValueError, "needs review"):
            prepare_dataset(self.source, self.output, min_images=1)
        self.assertFalse(self.output.exists())

    def test_minimum_is_not_silently_bypassed(self):
        seed_source(self.source)
        with self.assertRaisesRegex(ValueError, "at least 100"):
            prepare_dataset(self.source, self.output)

    def test_grouped_split_and_export_are_reproducible(self):
        seed_source(self.source, groups=6)
        first = prepare_dataset(self.source, self.output, min_images=1)
        second_path = self.root / "prepared_2"
        prepare_dataset(self.source, second_path, min_images=1)
        manifest = json.loads((self.output / "split_manifest.json").read_text())
        self.assertEqual(manifest, json.loads((second_path / "split_manifest.json").read_text()))
        for item in manifest["images"]:
            self.assertEqual(item["split"], manifest["sessions"][item["session"]])
        self.assertEqual(first["images"], 24)
        _, errors = validate_prepared(self.output, min_images=1)
        self.assertFalse(errors)
        archive = export_bundle(self.output, self.root / "bundle.zip", min_images=1)
        with zipfile.ZipFile(archive) as packaged:
            names = packaged.namelist()
            self.assertIn("semantic_dataset/core.py", names)
            self.assertIn("training/train_yolo.py", names)
            self.assertIn("screen_elements/data.yaml", names)
            self.assertNotIn("source/session_0/session.json", names)
        with self.assertRaises(FileExistsError):
            export_bundle(self.output, archive, min_images=1)

    def test_duplicate_sessions_block_until_excluded(self):
        seed_source(self.source)
        first = load_samples(self.source)[0]
        writer = SessionWriter(self.source, "duplicate_session", {})
        with Image.open(first.image) as image:
            path = writer.add(image)
        save_labels(writer.directory / "labels" / path.with_suffix(".txt").name, read_labels(first.label))
        _, _, errors = inspect_source(self.source)
        self.assertTrue(any("Duplicate screenshot" in error for error in errors))
        duplicate = next(sample for sample in load_samples(self.source) if sample.session == "duplicate_session")
        exclude_sample(duplicate)
        self.assertTrue(duplicate.image.exists())
        _, report, errors = inspect_source(self.source)
        self.assertFalse(errors)
        self.assertEqual(report["excluded_images"], 1)

    def test_changed_image_and_orphan_label_are_rejected(self):
        seed_source(self.source)
        sample = load_samples(self.source)[0]
        Image.new("RGB", (100, 60), "red").save(sample.image)
        _, _, errors = inspect_source(self.source)
        self.assertTrue(any("Image changed" in error for error in errors))
        (sample.label.parent / "orphan.txt").write_text("")
        with self.assertRaisesRegex(ValueError, "Orphan"):
            load_samples(self.source)

    def test_missing_class_in_groups_blocks_training_split(self):
        seed_source(self.source)
        for sample in load_samples(self.source):
            save_labels(sample.label, [Box(0, .5, .5, .2, .2)])
        with self.assertRaisesRegex(ValueError, "all three classes"):
            prepare_dataset(self.source, self.output, min_images=1)

    def test_prepared_missing_pair_and_tampered_class_names(self):
        seed_source(self.source)
        prepare_dataset(self.source, self.output, min_images=1)
        label = next((self.output / "labels/train").glob("*.txt"))
        label.rename(label.with_suffix(".missing"))
        _, errors = validate_prepared(self.output, min_images=1)
        self.assertTrue(any("pairing" in error for error in errors))
        (self.output / "data.yaml").write_text("names: {0: person}\n")
        _, errors = validate_prepared(self.output, min_images=1)
        self.assertTrue(any("three class IDs" in error for error in errors))

    def test_prepared_annotation_changes_are_detected(self):
        seed_source(self.source)
        prepare_dataset(self.source, self.output, min_images=1)
        label = next((self.output / "labels/train").glob("*.txt"))
        save_labels(label, [Box(0, .5, .5, .1, .1)])
        _, errors = validate_prepared(self.output, min_images=1)
        self.assertTrue(any("Annotation hash changed" in error for error in errors))

    def test_annotation_quality_hold_blocks_training_and_export(self):
        seed_source(self.source)
        prepare_dataset(self.source, self.output, min_images=1)
        write_json(self.output / "annotation_quality_hold.json", {
            "training_ready": False, "reason": "Camera boxes need visual review."})
        _, errors = validate_prepared(self.output, min_images=1)
        self.assertTrue(any("Annotation quality hold" in error for error in errors))
        with self.assertRaisesRegex(ValueError, "Annotation quality hold"):
            export_bundle(self.output, self.root / "bundle.zip", min_images=1)
        with self.assertRaisesRegex(ValueError, "Annotation quality hold"):
            train(self.output, self.root / "runs", min_images=1, dry_run=True)

    def test_prepared_session_leakage_and_path_escape_are_rejected(self):
        seed_source(self.source)
        prepare_dataset(self.source, self.output, min_images=1)
        path = self.output / "split_manifest.json"
        manifest = json.loads(path.read_text())
        train_item = next(item for item in manifest["images"] if item["split"] == "train")
        val_item = next(item for item in manifest["images"] if item["split"] == "val")
        val_item["session"] = train_item["session"]
        manifest["images"].append({"image": "../../outside.png", "label": "../../outside.txt", "split": "val"})
        write_json(path, manifest)
        _, errors = validate_prepared(self.output, min_images=1)
        self.assertTrue(any("Session leakage" in error for error in errors))
        self.assertTrue(any("Invalid manifest paths" in error for error in errors))

    def test_malformed_manifest_returns_diagnostic(self):
        self.output.mkdir()
        (self.output / "data.yaml").write_text("{}")
        (self.output / "split_manifest.json").write_text("[]")
        _, errors = validate_prepared(self.output)
        self.assertTrue(any("Malformed" in error for error in errors))

    def test_dry_run_needs_no_torch_and_does_not_create_runs(self):
        seed_source(self.source)
        prepare_dataset(self.source, self.output, min_images=1)
        run_path = self.root / "runs"
        result = train(self.output, run_path, min_images=1, dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["settings"]["fliplr"], 0)
        self.assertFalse(run_path.exists())

    def test_image_import_and_candidate_filter(self):
        folder = self.root / "screenshots"
        folder.mkdir()
        Image.new("RGB", (40, 20), "black").save(folder / "one.png")
        Image.new("RGB", (40, 20), "black").save(folder / "copy.png")
        result = import_images(folder, self.source, "import_01")
        self.assertEqual(result["saved"], 1)
        selection = CandidateFilter(.5)
        self.assertTrue(selection.accept(Image.new("RGB", (40, 20), "black")))
        self.assertFalse(selection.accept(Image.new("RGB", (40, 20), "black")))
        self.assertTrue(selection.accept(Image.new("RGB", (40, 20), "white")))
        with self.assertRaises(ValueError):
            validate_collection(3, 0, .5)

    def test_screen_collection_uses_rgb_and_writes_unreviewed_candidates(self):
        class Screen:
            def __init__(self, monitor):
                self.count = 0

            def __enter__(self):
                return self

            def grab(self):
                frame = np.zeros((60, 100, 3), dtype=np.uint8)
                frame[:, :, 2] = 80 + self.count * 60
                self.count += 1
                return frame

            def __exit__(self, *_):
                pass
        with patch("semantic_recorder.engine.ScreenSource", Screen):
            result = capture_screen(self.source, "screen_01", count=3, interval=.001, min_change=0)
        self.assertEqual(result["saved"], 3)
        samples = load_samples(self.source)
        with Image.open(samples[0].image) as image:
            self.assertEqual(image.getpixel((0, 0)), (80, 0, 0))
        self.assertFalse(any(sample.label.exists() for sample in samples))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required for video extraction fixture")
    def test_video_extraction_stops_at_end_and_saves_unique_candidates(self):
        from semantic_recorder.config import RecorderConfig
        from semantic_recorder.video import H264Writer, locate_ffmpeg
        video = self.root / "fixture.mp4"
        writer = H264Writer(locate_ffmpeg("ffmpeg"), video, 100, 60, RecorderConfig(video, audio="none"))
        for value in (30, 110, 220):
            writer.write(np.full((60, 100, 3), value, dtype=np.uint8))
        writer.close()
        result = extract_video(video, self.source, "video_01", count=10, interval=.1, min_change=0)
        self.assertEqual(result["saved"], 3)
        self.assertEqual([sample.metadata["source_seconds"] for sample in load_samples(self.source)], [0, .1, .2])

    def test_annotator_drag_coordinates_and_explicit_negative(self):
        import tkinter as tk
        from semantic_dataset.annotate import Annotator
        seed_source(self.source, groups=1, per_group=2, reviewed=False)
        root = tk.Tk()
        root.withdraw()
        try:
            app = Annotator(root, self.source)
            root.update_idletasks()
            app.canvas.configure(width=600, height=360)
            root.geometry("900x600")
            root.deiconify()
            root.update_idletasks()
            app.draw()
            x0, y0 = app.offset
            app.begin_box(SimpleNamespace(x=x0 + 10 * app.scale, y=y0 + 10 * app.scale))
            app.finish_box(SimpleNamespace(x=x0 + 60 * app.scale, y=y0 + 40 * app.scale))
            self.assertEqual(len(app.boxes), 1)
            self.assertAlmostEqual(app.boxes[0].x, .35)
            self.assertAlmostEqual(app.boxes[0].width, .5)
            app.save_next()
            self.assertTrue(app.samples[0].label.exists())
            self.assertFalse(app.samples[1].label.exists())
            app.negative_next()
            self.assertEqual(read_labels(app.samples[1].label), [])
        finally:
            root.destroy()


class NotebookTests(unittest.TestCase):
    def test_notebook_json_and_code_syntax(self):
        path = Path(__file__).resolve().parent.parent / "training/train_screen_elements_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def run_upload_cell(self, archive, workspace):
        notebook = json.loads((Path(__file__).resolve().parent.parent /
                               "training/train_screen_elements_colab.ipynb").read_text(encoding="utf-8"))
        upload = next("".join(cell["source"]) for cell in notebook["cells"]
                      if "uploaded = files.upload()" in "".join(cell["source"]))
        google = ModuleType("google")
        colab = ModuleType("google.colab")
        colab.files = SimpleNamespace(upload=lambda: {str(archive): b"mock browser upload"})
        namespace = {"sys": sys}
        previous_cwd, previous_path = os.getcwd(), sys.path.copy()
        try:
            with patch.dict(sys.modules, {"google": google, "google.colab": colab}), \
                 patch("tempfile.mkdtemp", return_value=str(workspace)):
                exec(compile(upload, "<colab-upload-cell>", "exec"), namespace)
            validation = next("".join(cell["source"]) for cell in notebook["cells"]
                              if "DATASET_REPORT, errors" in "".join(cell["source"]))
            exec(compile(validation, "<colab-validation-cell>", "exec"), namespace)
            return namespace
        finally:
            os.chdir(previous_cwd)
            sys.path[:] = previous_path

    def test_default_120_image_bundle_and_notebook_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, prepared = root / "source", root / "prepared"
            seed_source(source, groups=6, per_group=20)
            prepare_dataset(source, prepared)
            archive = export_bundle(prepared, root / "training_bundle.zip")
            workspace = root / "colab_workspace"
            workspace.mkdir()
            namespace = self.run_upload_cell(archive, workspace)
            self.assertEqual(namespace["DATASET_REPORT"]["images"], 120)
            self.assertTrue((workspace / "training/train_yolo.py").exists())

    def test_notebook_rejects_archive_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as packaged:
                packaged.writestr("../outside.txt", "should not be extracted")
            workspace = root / "workspace"
            workspace.mkdir()
            with self.assertRaisesRegex(AssertionError, "Unsafe archive path"):
                self.run_upload_cell(archive, workspace)
            self.assertFalse((root / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
