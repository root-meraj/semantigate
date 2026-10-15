from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile
import zipfile

from PIL import Image

from . import CLASSES


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def write_json(path: Path, value) -> None:
    """Atomically update metadata owned by this dataset workflow."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}-", suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def image_info(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.load()
    digest = hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()
    return image.width, image.height, digest


@dataclass(frozen=True)
class Box:
    class_id: int
    x: float
    y: float
    width: float
    height: float

    def validate(self) -> None:
        if type(self.class_id) is not int or not 0 <= self.class_id < len(CLASSES):
            raise ValueError("Class ID must be 0, 1, or 2.")
        values = (self.x, self.y, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Box coordinates must be finite.")
        if not (0 <= self.x <= 1 and 0 <= self.y <= 1 and 0 < self.width <= 1 and 0 < self.height <= 1):
            raise ValueError("YOLO centers must be in [0, 1]; widths/heights must be in (0, 1].")
        if (self.x - self.width / 2 < -1e-6 or self.x + self.width / 2 > 1 + 1e-6
                or self.y - self.height / 2 < -1e-6 or self.y + self.height / 2 > 1 + 1e-6):
            raise ValueError("The bounding box extends outside the image.")

    def line(self) -> str:
        self.validate()
        return f"{self.class_id} {self.x:.8f} {self.y:.8f} {self.width:.8f} {self.height:.8f}"


def read_labels(path: Path) -> list[Box]:
    if not path.is_file():
        raise ValueError(f"Unreviewed image: missing {path.name}. Mark a negative explicitly if there are no objects.")
    boxes = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            if len(parts) != 5 or not re.fullmatch(r"[0-9]+", parts[0]):
                raise ValueError("Expected class_id x_center y_center width height.")
            box = Box(int(parts[0]), *(float(value) for value in parts[1:]))
            box.validate()
            if box in seen:
                raise ValueError("Duplicate bounding box.")
        except ValueError as exc:
            raise ValueError(f"{path.name}:{number}: {exc}") from exc
        seen.add(box)
        boxes.append(box)
    return boxes


def save_labels(path: Path, boxes: list[Box], *, negative=False) -> None:
    if not boxes and not negative:
        raise ValueError("Use 'Mark negative' to review an image with no target objects.")
    lines = [box.line() for box in boxes]
    if len(set(lines)) != len(lines):
        raise ValueError("Remove duplicate boxes before saving.")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + ("\n" if lines else "")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}-", delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


@dataclass(frozen=True)
class Sample:
    session: str
    image: Path
    label: Path
    manifest: Path
    metadata: dict

    @property
    def excluded(self):
        return self.metadata.get("excluded", False)


def valid_session_name(name: str) -> None:
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) or name.upper() in reserved:
        raise ValueError("Session names must use 1–64 letters/digits/hyphens/underscores and avoid Windows device names.")


class SessionWriter:
    def __init__(self, root: Path, session: str, source: dict):
        valid_session_name(session)
        self.directory = root.expanduser().resolve() / session
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "images").mkdir()
        (self.directory / "labels").mkdir()
        self.manifest = self.directory / "session.json"
        self.data = {"schema_version": 1, "session": session, "classes": list(CLASSES),
                     "created_at": datetime.now().astimezone().isoformat(), "source": source, "samples": []}
        self._hashes = set()
        write_json(self.manifest, self.data)

    def add(self, image: Image.Image, seconds: float | None = None) -> Path | None:
        image = image.convert("RGB")
        digest = hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()
        if digest in self._hashes:
            return None
        name = f"{self.data['session']}_{len(self.data['samples']) + 1:05d}.png"
        path = self.directory / "images" / name
        with path.open("xb") as handle:
            image.save(handle, format="PNG")
        self.data["samples"].append({"image": name, "width": image.width, "height": image.height,
                                     "pixel_sha256": digest, "source_seconds": seconds, "excluded": False})
        write_json(self.manifest, self.data)
        self._hashes.add(digest)
        return path


def load_samples(root: Path) -> list[Sample]:
    samples = []
    for manifest in sorted(root.expanduser().resolve().glob("*/session.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or data.get("schema_version") != 1
                or data.get("classes") != list(CLASSES) or not isinstance(data.get("samples"), list)):
            raise ValueError(f"Unsupported schema or class mapping in {manifest}.")
        if data.get("session") != manifest.parent.name:
            raise ValueError(f"Session name/folder mismatch in {manifest}.")
        valid_session_name(data["session"])
        names = set()
        for item in data["samples"]:
            if not isinstance(item, dict) or not isinstance(item.get("image"), str) or type(item.get("excluded", False)) is not bool:
                raise ValueError(f"Malformed sample entry in {manifest}.")
            name = item["image"]
            if (Path(name).name != name
                    or not re.fullmatch(re.escape(data["session"]) + r"_[0-9]{5,}\.png", name) or name in names):
                raise ValueError(f"Invalid or duplicate sample filename in {manifest}.")
            names.add(name)
            samples.append(Sample(data["session"], manifest.parent / "images" / name,
                                  manifest.parent / "labels" / Path(name).with_suffix(".txt"), manifest, item))
        actual_images = {path.name for path in (manifest.parent / "images").iterdir() if path.is_file()}
        if actual_images != names:
            raise ValueError(f"Images and session manifest differ in {manifest.parent}; import edited/new files as a new session.")
        unexpected_labels = {path.name for path in (manifest.parent / "labels").glob("*.txt")} - {
            Path(name).with_suffix(".txt").name for name in names}
        if unexpected_labels:
            raise ValueError(f"Orphan labels in {manifest.parent}: {sorted(unexpected_labels)}")
    if not samples:
        raise ValueError("No samples found. Capture or import screenshots first.")
    return samples


def exclude_sample(sample: Sample, excluded=True) -> None:
    data = json.loads(sample.manifest.read_text(encoding="utf-8"))
    item = next(item for item in data["samples"] if item["image"] == sample.image.name)
    item["excluded"] = bool(excluded)
    write_json(sample.manifest, data)
    sample.metadata["excluded"] = bool(excluded)


def inspect_source(root: Path) -> tuple[list[Sample], dict, list[str]]:
    all_samples = load_samples(root)
    samples = [sample for sample in all_samples if not sample.excluded]
    counts = Counter()
    groups = Counter()
    hashes = {}
    errors = []
    negatives = reviewed = 0
    for sample in samples:
        groups[sample.session] += 1
        try:
            width, height, digest = image_info(sample.image)
            if (width, height, digest) != (sample.metadata["width"], sample.metadata["height"], sample.metadata["pixel_sha256"]):
                raise ValueError("Image changed after collection; import the edited screenshot as a new session.")
            if digest in hashes:
                raise ValueError(f"Duplicate screenshot also exists at {hashes[digest]}; exclude one copy.")
            hashes[digest] = sample.image
            boxes = read_labels(sample.label)
            reviewed += 1
            negatives += not boxes
            counts.update(box.class_id for box in boxes)
        except (OSError, ValueError) as exc:
            errors.append(f"{sample.session}/{sample.image.name}: {exc}")
    report = {"included_images": len(samples), "excluded_images": len(all_samples) - len(samples),
              "reviewed_images": reviewed, "negative_images": negatives, "sessions": dict(groups),
              "instances": {name: counts[index] for index, name in enumerate(CLASSES)}}
    return samples, report, errors


def plan_splits(samples: list[Sample], val_fraction=0.15, test_fraction=0.15, seed=42) -> dict[str, str]:
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than 1.")
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample.session, []).append(sample)
    names = sorted(grouped)
    if len(names) < 3:
        raise ValueError("At least 3 independent sessions are needed for train/validation/test. Aim for 6 or more.")
    val_count = max(1, round(len(names) * val_fraction))
    test_count = max(1, round(len(names) * test_fraction))
    train_count = len(names) - val_count - test_count
    if train_count < 1:
        raise ValueError("The selected fractions leave no training sessions.")
    classes = {name: {box.class_id for sample in group for box in read_labels(sample.label)}
               for name, group in grouped.items()}
    targets = {"train": 1 - val_fraction - test_fraction, "val": val_fraction, "test": test_fraction}
    best = None
    best_score = float("inf")
    rng = random.Random(seed)
    for _ in range(2000):
        shuffled = names.copy()
        rng.shuffle(shuffled)
        plan = {name: ("train" if i < train_count else "val" if i < train_count + val_count else "test")
                for i, name in enumerate(shuffled)}
        present = {split: set().union(*(classes[name] for name in names if plan[name] == split)) for split in SPLITS}
        if any(present[split] != set(range(len(CLASSES))) for split in SPLITS):
            continue
        counts = Counter()
        for name in names:
            counts[plan[name]] += len(grouped[name])
        score = sum((counts[split] / len(samples) - targets[split]) ** 2 for split in SPLITS)
        if score < best_score:
            best, best_score = plan, score
    if best is None:
        raise ValueError("No split with all three classes in train/val/test was found. Collect more varied independent sessions.")
    return dict(sorted(best.items()))


def prepare_dataset(source: Path, output: Path, *, min_images=100, val_fraction=0.15, test_fraction=0.15, seed=42) -> dict:
    import yaml
    if min_images < 1:
        raise ValueError("Minimum image count must be positive.")
    samples, report, errors = inspect_source(source)
    if errors:
        raise ValueError("Dataset needs review:\n" + "\n".join(errors[:15]) +
                         (f"\n…and {len(errors) - 15} more." if len(errors) > 15 else ""))
    if len(samples) < min_images:
        raise ValueError(f"Only {len(samples)} reviewed images; at least {min_images} are required. The MVP target is 100–200.")
    plan = plan_splits(samples, val_fraction, test_fraction, seed)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    for sample in samples:
        split = plan[sample.session]
        image = Path("images") / split / sample.image.name
        label = Path("labels") / split / sample.label.name
        shutil.copyfile(sample.image, output / image)
        shutil.copyfile(sample.label, output / label)
        records.append({"image": image.as_posix(), "label": label.as_posix(), "split": split,
                        "session": sample.session, "pixel_sha256": sample.metadata["pixel_sha256"],
                        "label_sha256": hashlib.sha256(sample.label.read_bytes()).hexdigest()})
    config = {"path": ".", "train": "images/train", "val": "images/val", "test": "images/test",
              "names": dict(enumerate(CLASSES))}
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(output / "split_manifest.json", {"schema_version": 1, "seed": seed, "sessions": plan,
               "requested_fractions": {"train": 1 - val_fraction - test_fraction, "val": val_fraction, "test": test_fraction},
               "images": records})
    result, errors = validate_prepared(output, min_images=min_images)
    if errors:
        raise ValueError("Prepared data did not pass validation:\n" + "\n".join(errors))
    result["source_summary"] = report
    write_json(output / "dataset_report.json", result)
    return result


def validate_prepared(root: Path, *, min_images=100) -> tuple[dict, list[str]]:
    """Validate this project's packaged format, including session/hash leakage."""
    import yaml
    root = root.expanduser().resolve()
    errors = []
    if min_images < 1:
        return {}, ["Minimum image count must be positive."]
    quality_hold = root / "annotation_quality_hold.json"
    if quality_hold.exists():
        try:
            review = json.loads(quality_hold.read_text(encoding="utf-8"))
            if not isinstance(review, dict) or review.get("training_ready") is not True:
                reason = review.get("reason", "Visual annotation review remains incomplete.") if isinstance(review, dict) else "Invalid quality-review record."
                errors.append(f"Annotation quality hold: {reason}")
        except (OSError, ValueError) as exc:
            errors.append(f"Cannot read annotation quality hold: {exc}")
    try:
        config = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
        manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return {}, [f"Cannot read dataset configuration/manifest: {exc}"]
    if (not isinstance(manifest, dict) or not isinstance(manifest.get("images"), list)
            or not isinstance(manifest.get("sessions"), dict)
            or not isinstance(manifest.get("requested_fractions"), dict)):
        return {}, ["Malformed split manifest; expected image records and session assignments."]
    if not isinstance(config, dict) or config.get("names") != dict(enumerate(CLASSES)):
        errors.append("data.yaml must use exactly the project's three class IDs and names.")
    if manifest.get("schema_version") != 1:
        errors.append("Unsupported split-manifest schema.")
    counts = {split: {"images": 0, "negative_images": 0, "instances": Counter()} for split in SPLITS}
    records = {}
    groups = {}
    for item in manifest.get("images", []):
        if not isinstance(item, dict):
            errors.append("Malformed image entry in split manifest.")
            continue
        split = item.get("split")
        image = item.get("image", "")
        label = item.get("label", "")
        if (not isinstance(image, str) or not isinstance(label, str) or split not in SPLITS
                or not re.fullmatch(r"images/(train|val|test)/[A-Za-z0-9_-]+\.png", image)
                or image.split("/")[1] != split
                or label != image.replace("images/", "labels/", 1).removesuffix(".png") + ".txt"):
            errors.append(f"Invalid manifest paths: {image} / {label}")
            continue
        if image in records:
            errors.append(f"Duplicate image entry: {image}")
        records[image] = item
        session = item.get("session", "")
        if not isinstance(session, str) or not session:
            errors.append(f"Missing session for {image}")
            continue
        if manifest.get("sessions", {}).get(session) != split:
            errors.append(f"Missing/inconsistent session assignment for {image}")
        if session in groups and groups[session] != split:
            errors.append(f"Session leakage across splits: {session}")
        groups[session] = split
    hashes = {}
    actual = set()
    for split in SPLITS:
        if isinstance(config, dict) and config.get(split) != f"images/{split}":
            errors.append(f"Unexpected {split} path in data.yaml.")
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"Missing images/labels directory for {split}.")
            continue
        images = sorted(path for path in image_dir.iterdir() if path.is_file())
        label_stems = {path.stem for path in label_dir.glob("*.txt")}
        if label_stems != {path.stem for path in images}:
            errors.append(f"Image/label pairing mismatch in {split}.")
        for image in images:
            relative = image.relative_to(root).as_posix()
            actual.add(relative)
            if relative not in records:
                errors.append(f"Unmanifested image: {relative}")
            try:
                width, height, digest = image_info(image)
                if relative in records and digest != records[relative].get("pixel_sha256"):
                    raise ValueError("Image hash changed after preparation.")
                if digest in hashes:
                    raise ValueError(f"Duplicate image pixels also found at {hashes[digest]} (possible leakage).")
                hashes[digest] = relative
                label_path = label_dir / image.with_suffix(".txt").name
                boxes = read_labels(label_path)
                if (relative in records and hashlib.sha256(label_path.read_bytes()).hexdigest()
                        != records[relative].get("label_sha256")):
                    raise ValueError("Annotation hash changed after preparation; rebuild from reviewed source labels.")
                counts[split]["images"] += 1
                counts[split]["negative_images"] += not boxes
                counts[split]["instances"].update(box.class_id for box in boxes)
            except (OSError, ValueError) as exc:
                errors.append(f"{relative}: {exc}")
        for index, name in enumerate(CLASSES):
            if not counts[split]["instances"][index]:
                errors.append(f"No {name} instances in {split}.")
    for missing in sorted(set(records) - actual):
        errors.append(f"Manifest image is missing: {missing}")
    total = sum(value["images"] for value in counts.values())
    if total < min_images:
        errors.append(f"Only {total} valid reviewed images; need at least {min_images}.")
    result = {"images": total, "sessions": len(groups), "classes": list(CLASSES), "splits": {}, "warnings": []}
    for split in SPLITS:
        value = counts[split]
        result["splits"][split] = {"images": value["images"], "negative_images": value["negative_images"],
            "instances": {name: value["instances"][index] for index, name in enumerate(CLASSES)}}
        requested = manifest.get("requested_fractions", {}).get(split)
        if requested is not None and (not isinstance(requested, (int, float)) or not math.isfinite(requested)):
            errors.append(f"Invalid requested fraction for {split}.")
        elif requested is not None and total and abs(value["images"] / total - requested) > 0.15:
            result["warnings"].append(f"{split} differs from the requested fraction; group isolation takes priority.")
    if len(groups) < 6:
        result["warnings"].append("Fewer than 6 independent sessions; add diversity before drawing performance conclusions.")
    return result, errors


def export_bundle(root: Path, destination: Path, *, min_images=100) -> Path:
    report, errors = validate_prepared(root, min_images=min_images)
    if errors:
        raise ValueError("Cannot export invalid data:\n" + "\n".join(errors[:15]))
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        files = [root / "data.yaml", root / "split_manifest.json"]
        files += sorted((root / "images").glob("*/*.png")) + sorted((root / "labels").glob("*/*.txt"))
        for path in files:
            archive.write(path, "screen_elements/" + path.relative_to(root).as_posix())
        archive.writestr("screen_elements/dataset_report.json", json.dumps(report, indent=2))
        for name in ("__init__.py", "core.py"):
            path = Path(__file__).parent / name
            archive.write(path, f"semantic_dataset/{name}")
        project = Path(__file__).resolve().parent.parent
        for relative in ("training/__init__.py", "training/train_yolo.py", "requirements-yolo.txt"):
            archive.write(project / relative, relative)
    return destination
