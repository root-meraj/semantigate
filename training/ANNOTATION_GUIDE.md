# Screen-element annotation guide

Use the same class IDs for collection, annotation, training, and inference.
Draw boxes around visible content regions, rather than faces, people, or screen text.

| ID | Class | Include | Exclude |
| --- | --- | --- | --- |
| 0 | `Speaker_Video` | Each visible participant camera tile, including a floating camera overlay | Avatar-only tiles, camera-off placeholders, name-only labels |
| 1 | `Presentation_Slide` | The visible slide canvas, in presentation mode or inside a shared slide application | Slide thumbnails, toolbars, unrelated application chrome |
| 2 | `Screen_Share` | Shared non-slide application/desktop/browser content | Ordinary conference UI, speaker tiles, slide canvases |

A slide shown through screen sharing is `Presentation_Slide`. Do not give that
same content a second `Screen_Share` box. Speaker video floating over a slide
can have its own overlapping box because it is a separate content region. Use
the visible tile/canvas boundaries; crop boxes to the image edges if partially
visible. Review every visible instance, including small camera tiles.
Ignore recursive copies inside a shared-screen preview and slide thumbnails;
they are previews of existing content, not additional live participant tiles or
main slide canvases. A document/PDF page is not automatically a presentation slide.

For an ambiguous full-screen image, use session context to establish whether it
is a shared application, slide, or ordinary local window. Exclude the sample
when its meaning cannot be established. A bounding-box detector cannot reliably
infer "shared" from pixels when a shared browser and a local browser look identical;
include realistic meeting UI/context and test these failure cases explicitly.
An open Meet tab or camera overlay alone does not prove screen sharing is active.
Use visible sharing indicators, such as a Stop sharing banner and a matching
presentation preview, or verified session context. Combine adjacent capture
batches from the same ongoing call into one group, regardless of folder names.

## Collect a useful pilot dataset

Aim initially for **100–200 reviewed screenshots** across **at least six independent
sessions**. Three sessions are the tool's hard minimum. Aim for at least 30 images
containing each class; images may contain multiple classes. Add genuine negative
examples so conference controls or blank screens do not become false positives.
These are pilot targets, not evidence that the model will generalize.

Vary meeting applications, layouts, light/dark themes, display resolution, camera
tile sizes, slide styles, gallery/speaker views, shared browsers/editors, and
transitions between content types. Capture actual meeting views or deliberate
staged meetings. Random test-fixture pictures do not count as training examples.

Use a session ID for one recording/meeting, including any clips or screenshot
folders taken from it. Combine those screenshots in one import folder/session;
splitting clips into separate IDs would defeat the group split. Collect independent
sessions rather than filling train/validation/test with adjacent screenshots.
Prefer original screenshots or ungated source recordings: frozen/compressed
outputs underrepresent the changes the final retention policy must preserve.

Remove or anonymize sensitive meeting content **before import**, then annotate
the final images. Source manifests track decoded image hashes; editing an image
after annotation requires importing the edited image as a new session and reviewing
its boxes again. Collection stays local; Colab upload happens when you run the
notebook's upload cell.

## Use the local annotator

Run `python -m semantic_dataset annotate`, or double-click `run_annotator.bat`
after collection. Choose a class, then drag a rectangle over each matching region.

- **Save & next / S:** save boxes and continue.
- **Mark negative & next / N:** explicitly review a frame with none of the classes.
  Remove any boxes first.
- **Delete selected / Undo U:** correct a box, then draw its replacement.
- **Exclude from dataset:** omit an ambiguous/duplicate/irrelevant frame without
  deleting the raw image. The same button includes it again.
- **Left/right:** navigate. **1/2/3:** select the three classes.

Missing label files mean unreviewed images. Empty files mean reviewed negatives.
Saving boxes and marking negatives are separate actions to avoid silently using
unreviewed screenshots as negatives. The preparation command refuses unfinished
included images, malformed boxes, duplicate screenshots, and missing class coverage.

## Evaluate before changing the recorder gate

The training notebook starts from YOLO11n and records validation metrics, per-class
mAP50–95, inference timing, dataset hashes, and checkpoint hashes. Review boundary
errors and missed small tiles in validation predictions. Finalize settings using
validation, then evaluate the reserved test sessions once.

After detection works, define a region retention policy and evaluate it on an
unseen **time-ordered recording**. Measure missed slide changes, delayed small
edits, preserved speaker movement, text readability, elapsed duration, audio
alignment, encoder/inference cost, and file bytes versus the existing pixel gate
and ungated H.264. Detection accuracy alone does not prove compression savings.
