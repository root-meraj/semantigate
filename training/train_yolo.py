import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path

from semantic_dataset import CLASSES
from semantic_dataset.core import validate_prepared, write_json


ULTRALYTICS_VERSION = "8.4.175"


def training_settings(*, model="yolo11n.pt", epochs=100, imgsz=640, batch=16, device="0", seed=42):
    if epochs < 1 or batch < 1 or imgsz < 32 or imgsz % 32:
        raise ValueError("Epochs/batch must be positive and image size a positive multiple of 32.")
    return {"model": model, "epochs": epochs, "imgsz": imgsz, "batch": batch, "device": device,
            "seed": seed, "deterministic": True, "patience": 20, "workers": 2,
            "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "shear": 0.0, "perspective": 0.0,
            "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.1, "translate": 0.03, "scale": 0.1}


def train(data: Path, project: Path, *, model="yolo11n.pt", epochs=100, imgsz=640, batch=16,
          device="0", seed=42, evaluate_test=False, dry_run=False, min_images=100) -> dict:
    import yaml
    data = data.expanduser().resolve()
    project = project.expanduser().resolve()
    report, errors = validate_prepared(data, min_images=min_images)
    if errors:
        raise ValueError("Training blocked by dataset validation:\n" + "\n".join(errors[:20]))
    settings = training_settings(model=model, epochs=epochs, imgsz=imgsz, batch=batch, device=device, seed=seed)
    if dry_run:
        return {"dry_run": True, "dataset": report, "settings": settings, "ultralytics_version": ULTRALYTICS_VERSION}
    if importlib.metadata.version("ultralytics") != ULTRALYTICS_VERSION:
        raise ValueError(f"Use ultralytics=={ULTRALYTICS_VERSION} from requirements-yolo.txt for this training recipe.")
    import torch
    from ultralytics import YOLO
    if device != "cpu" and not torch.cuda.is_available():
        raise ValueError("No CUDA GPU is available. Select a GPU in Colab or explicitly choose --device cpu.")
    project.mkdir(parents=True, exist_ok=True)
    name = f"screen_elements_{datetime.now():%Y%m%d_%H%M%S_%f}"
    recipe_dir = project / f"{name}_recipe"
    recipe_dir.mkdir(exist_ok=False)
    config = {"path": str(data), "train": "images/train", "val": "images/val", "test": "images/test",
              "names": dict(enumerate(CLASSES))}
    yaml_path = recipe_dir / "runtime_data.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    model_object = YOLO(model)
    arguments = {key: value for key, value in settings.items() if key != "model"}
    model_object.train(data=str(yaml_path), project=str(project), name=name, exist_ok=False, **arguments)
    checkpoint = Path(model_object.trainer.best).resolve()
    best = YOLO(str(checkpoint))
    if dict(best.names) != dict(enumerate(CLASSES)):
        raise ValueError("The trained checkpoint has an unexpected class mapping.")
    validation = best.val(data=str(yaml_path), split="val", imgsz=imgsz, batch=batch,
                          device=device, project=str(project), name=f"{name}_validation")
    result = {"checkpoint": str(checkpoint), "classes": list(CLASSES), "settings": settings,
              "ultralytics_version": ULTRALYTICS_VERSION, "torch_version": str(torch.__version__),
              "dataset": report, "validation": dict(validation.results_dict),
              "validation_map50_95_per_class": {name: float(validation.box.maps[index])
                                                for index, name in enumerate(CLASSES)},
              "validation_speed_ms": dict(validation.speed), "held_out_test": None,
              "split_manifest_sha256": hashlib.sha256((data / "split_manifest.json").read_bytes()).hexdigest(),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    if evaluate_test:
        metrics = best.val(data=str(yaml_path), split="test", imgsz=imgsz, batch=batch, device=device,
                           project=str(project), name=f"{name}_test")
        result["held_out_test"] = dict(metrics.results_dict)
    write_json(checkpoint.parent.parent / "training_report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the three-class screen-element detector on reviewed, grouped data.")
    parser.add_argument("--data", type=Path, default=Path("datasets/screen_elements/prepared"))
    parser.add_argument("--project", type=Path, default=Path("runs/screen_elements"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-images", type=int, default=100, help="Lower only for tooling smoke tests")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate test only after model/settings are finalized")
    parser.add_argument("--dry-run", action="store_true", help="Validate data/recipe without importing Torch or downloading weights")
    args = parser.parse_args(argv)
    try:
        result = train(args.data, args.project, model=args.model, epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, device=args.device, seed=args.seed, min_images=args.min_images,
            evaluate_test=args.evaluate_test, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
