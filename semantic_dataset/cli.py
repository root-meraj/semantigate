import argparse
import json
from pathlib import Path
import sys

from .core import export_bundle, inspect_source, prepare_dataset, validate_prepared


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Collect, annotate, validate, and package screen-element YOLO data.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "extract", "import"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, default=Path("datasets/screen_elements/source"))
        command.add_argument("--session", required=True, help="Unique recording/session identifier; whole sessions stay in one split")
        if name == "import":
            command.add_argument("--images", type=Path, required=True)
        else:
            command.add_argument("--count", type=int, default=40, help="Maximum capture/sample attempts")
            command.add_argument("--interval", type=float, default=3)
            command.add_argument("--min-change", type=float, default=0.5, help="Mean grayscale difference on a 160x90 thumbnail")
            if name == "extract":
                command.add_argument("--video", type=Path, required=True)
            else:
                command.add_argument("--monitor", type=int)
    for name in ("annotate", "status"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, default=Path("datasets/screen_elements/source"))
    prepare = commands.add_parser("prepare", help="Validate reviewed data and split whole sessions")
    prepare.add_argument("--source", type=Path, default=Path("datasets/screen_elements/source"))
    prepare.add_argument("--output", type=Path, default=Path("datasets/screen_elements/prepared"))
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--val-fraction", type=float, default=0.15)
    prepare.add_argument("--test-fraction", type=float, default=0.15)
    prepare.add_argument("--min-images", type=int, default=100, help="Lower only for tooling smoke tests")
    for name in ("validate", "export"):
        command = commands.add_parser(name)
        command.add_argument("--dataset", type=Path, default=Path("datasets/screen_elements/prepared"))
        command.add_argument("--min-images", type=int, default=100)
        if name == "export":
            command.add_argument("--output", type=Path, default=Path("datasets/screen_elements/training_bundle.zip"))
    args = parser.parse_args(argv)
    try:
        if args.command in {"capture", "extract", "import"}:
            from .collect import capture_screen, extract_video, import_images
            if args.command == "import":
                result = import_images(args.images, args.root, args.session)
            else:
                settings = {"count": args.count, "interval": args.interval, "min_change": args.min_change}
                result = (extract_video(args.video, args.root, args.session, **settings) if args.command == "extract"
                          else capture_screen(args.root, args.session, monitor=args.monitor, **settings))
            print(json.dumps(result, indent=2))
        elif args.command == "annotate":
            from .annotate import launch
            launch(args.root)
        elif args.command == "status":
            _, report, errors = inspect_source(args.root)
            print(json.dumps(report, indent=2))
            for error in errors[:15]:
                print(f"Review needed: {error}")
            if len(errors) > 15:
                print(f"...and {len(errors) - 15} more images needing review.")
        elif args.command == "prepare":
            report = prepare_dataset(args.source, args.output, min_images=args.min_images, seed=args.seed,
                val_fraction=args.val_fraction, test_fraction=args.test_fraction)
            print(json.dumps(report, indent=2))
        elif args.command == "validate":
            report, errors = validate_prepared(args.dataset, min_images=args.min_images)
            print(json.dumps(report, indent=2))
            if errors:
                raise ValueError("\n".join(errors[:20]))
            print("PASS: dataset labels, image pairs, class coverage, hashes, and session separation.")
        elif args.command == "export":
            print(export_bundle(args.dataset, args.output, min_images=args.min_images))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
