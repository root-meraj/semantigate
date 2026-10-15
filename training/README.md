# YOLO dataset and Colab workflow

The collection/annotation/packaging tools are ready. A trained screen-element
model still requires real screenshots with visually reviewed labels. The recorder
continues to use its existing pixel gate while that dataset/model is evaluated.

## 1. Install the lightweight local tools

```powershell
cd B:\motion_gate
.\venv\Scripts\python.exe -m pip install -r requirements-dataset.txt
```

Read [the annotation guide](ANNOTATION_GUIDE.md) before collecting or drawing boxes.

## 2. Collect candidates from independent sessions

Choose one source for each session. Use meaningful unique IDs such as `zoom_01`,
`meet_02`, or `slides_03`. The capture count limits attempts; static/duplicate
screenshots are filtered, so fewer than that count may be saved.

```powershell
# Capture the primary monitor every three seconds, at most 40 attempts.
# Run while displaying the meeting content you want to label. Ctrl+C ends collection.
.\venv\Scripts\python.exe -m semantic_dataset capture --session zoom_01 --count 40 --interval 3

# Extract candidate frames from an original, ungated recording.
.\venv\Scripts\python.exe -m semantic_dataset extract --session meet_02 --video "B:\source_videos\meeting.mp4" --count 40 --interval 3

# Import an existing folder of screenshots from one session.
.\venv\Scripts\python.exe -m semantic_dataset import --session slides_03 --images "B:\source_screenshots\slides"
```

The example source paths must be replaced with your actual files. Each session
goes into `datasets/screen_elements/source/<session>/` with images, a label folder,
and collection metadata. Existing sessions are protected from replacement.
Reuse one session grouping for all clips/screenshots from the same meeting:
combine their images into one folder before import. Add at least six varied
sessions and aim for 100–200 **included, reviewed** images after filtering.

`--min-change 0` disables thumbnail change filtering; exact duplicate pixels are
still skipped. `--monitor` selects a screen. Import converts supported images to
lossless RGB PNG and removes exact duplicates within the session.

## 3. Label and review

```powershell
.\venv\Scripts\python.exe -m semantic_dataset annotate
.\venv\Scripts\python.exe -m semantic_dataset status
```

The Tkinter annotator draws YOLO boxes, saves positives, explicitly marks negatives,
and excludes unsuitable candidates. `status` reports review progress and problems.
Use the annotation guide's mutually exclusive slide/shared-app rule.

## 4. Prepare, validate, and export

```powershell
.\venv\Scripts\python.exe -m semantic_dataset prepare
.\venv\Scripts\python.exe -m semantic_dataset validate
.\venv\Scripts\python.exe -m semantic_dataset export
```

Preparation defaults to 100 images minimum and approximate 70/15/15 train/val/test
fractions, using whole sessions and seed 42. It searches seeded session assignments
for class coverage and close image counts; it never splits a session to force a
ratio. Collect more diverse groups if it cannot find adequate class coverage.
Ratio/diversity warnings appear in the report. Exact duplicate screenshots across
sessions must be excluded. Near duplicates are not guaranteed to be detected.
Image and annotation hashes detect changes after preparation; rebuild a dataset
version from reviewed source labels when annotations change.

Outputs are `datasets/screen_elements/prepared/` and
`datasets/screen_elements/training_bundle.zip`. The ZIP includes the reviewed
images/labels, class YAML, split manifest, fresh validation report, and project
validation/training helpers. Existing prepared folders and ZIPs are protected;
use a new `--output` path for another version. `--min-images` may be lowered for
tooling checks, but the Colab notebook retains the 100-image pilot requirement.

## 5. Train in Colab

Open **train_screen_elements_colab.ipynb** in Google Colab, select a GPU runtime,
and run its cells. Upload the generated ZIP when prompted. The notebook validates
the data, fine-tunes YOLO11n, reports validation and per-class metrics, shows
predictions, and downloads `best.pt` with the recorded evidence. The held-out test
cell defaults to disabled until model/settings have been selected using validation.

The recipe pins Ultralytics 8.4.175, defaults to 100 epochs, image size 640, batch
16, patience 20, and disables flips/rotations/mosaic for text-bearing screenshots.
This is a baseline recipe, not a claim of trained accuracy. GPU training has not
been executed yet. On 14 October 2026, the agent visually reviewed all 148 real
screenshots and saved corrected annotations in
`datasets/screen_elements/source_reviewed_v2/`. There are 126 included images,
22 excluded transitions, 48 camera boxes, 14 slide boxes, and 15 confirmed
non-slide sharing boxes. No further manual annotation of these captures is needed.

Only one call contains confirmed active non-slide sharing. Original capture
batches 03–05 show the same continuous Meet call and were grouped together.
The corrected source therefore cannot yet form train/validation/test splits
with all three classes in each split. Two additional independent practice calls
with camera, slides, and actual browser/document sharing close this specific gap.
See the [completed review and capture instructions](../datasets/screen_elements/annotation_review/full_review/RESULT.md).

The reviewed-source ZIP is a source backup, not the Colab training ZIP. Keep the
older `prepared` snapshot blocked by its `annotation_quality_hold.json`. Once the
new captures have been annotated, prepare a fresh folder using the reviewed root:

```powershell
.\venv\Scripts\python.exe -m semantic_dataset status --root datasets\screen_elements\source_reviewed_v2
.\venv\Scripts\python.exe -m semantic_dataset prepare --source datasets\screen_elements\source_reviewed_v2 --output datasets\screen_elements\prepared_v2
.\venv\Scripts\python.exe -m semantic_dataset validate --dataset datasets\screen_elements\prepared_v2
.\venv\Scripts\python.exe -m semantic_dataset export --dataset datasets\screen_elements\prepared_v2 --output datasets\screen_elements\training_bundle_v2.zip
```

Optional local training should use a separate Python environment and
`requirements-yolo.txt`. The lightweight dry run works in the existing environment
without importing Torch or downloading model weights:

```powershell
.\venv\Scripts\python.exe -m training.train_yolo --dry-run

# In the separate GPU-training environment:
python -m training.train_yolo --data datasets\screen_elements\prepared --device 0
```

Training and Colab both rewrite the dataset root into a runtime YAML, so the same
prepared archive works across Windows and Linux. Place the downloaded checkpoint
in `models/` for the next retention-policy milestone.

## Checks performed locally

The test suite exercises labels, explicit negatives, annotation coordinates,
session grouping, image changes/duplicates, class coverage, minimum image counts,
protected outputs, ZIP packaging, notebook syntax, archive extraction/validation,
and a training dry run. The full pipeline is also exercised with 120 synthetic
fixture images; those fixtures test tooling and are not supplied as model training
data. Run all recorder and dataset regressions with:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Format/API references: [Ultralytics detection datasets](https://docs.ultralytics.com/datasets/detect),
[YOLO11](https://docs.ultralytics.com/models/yolo11),
[training](https://docs.ultralytics.com/modes/train), and
[validation](https://docs.ultralytics.com/modes/val).
