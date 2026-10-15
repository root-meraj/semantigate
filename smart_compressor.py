"""Launch the Phase 3 recorder (GUI by default, --cli for terminal recording)."""

from semantic_recorder.cli import main


def start_timeline_compressor():
    return main()


if __name__ == "__main__":
    raise SystemExit(start_timeline_compressor())
