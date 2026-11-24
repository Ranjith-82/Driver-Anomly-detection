# Offline video evaluation

This package reads recorded video frames with OpenCV and calls the existing
Driver Monitor detector adapters and runtime methods. It does not open a webcam,
start live worker threads, or modify model files, thresholds, or alert behavior.
Each clip is processed sequentially with frame timestamps in video seconds so
stateful behavior, 1.5-second drowsiness, and gaze persistence can advance using
the recorded clip timeline.

## Existing runtime mapping and constraints

| Evaluation task | Existing inference path | Offline/state requirements |
|---|---|---|
| `behavior`, `gaze`, `fatigue` | `BehaviorWorker.process_packet` from `src/driver_monitor/monitoring/legacy_runtime.py`, which delegates to the existing behavior/gaze adapters | Needs the behavior model, labels, face-landmarker model, and a saved gaze-calibration JSON. The existing head-pose estimator performs its normal 45-frame neutral-pose calibration at each clip start; those initial frames are reported as `CALIBRATING`. Eye closure, gaze smoothing/distraction, and behavior persistence receive each frame's video timestamp. |
| `yolo` | `YoloObjectDetector.detect` from `src/driver_monitor/detection/objects.py` | Needs the existing object weights and complete labels for every decoded frame. Reports low-floor raw boxes and live class-confidence-filtered boxes separately; live phone-use confirmation requires driver identity/face labels and is reported unavailable otherwise. |
| `recognition` | `recognize_frame` from `src/driver_monitor/detection/recognition.py` | Needs the existing local driver database/photos and face-recognition dependency. Each frame is scored with the same HOG face locations, encoding function, tolerance, and resize path. Worker confirmation scheduling is intentionally not run. Use one driver face per frame. |
| `emotion` | `SVMEmotionAnalyzer.predict` and the existing `SVMEmotionWorker._crop_face` helper | Needs the packaged SVM artifacts and MediaPipe. It evaluates model output independently of the live requirement that the selected face belongs to a confirmed driver. DeepFace threshold-gated inference is not included in this SVM task. |

No `CameraProducer`, webcam calibration UI, detection worker thread, alert
manager, dashboard, or worker frame scheduler is started. Analyzer state is
created fresh for each video. For behavior and gaze results, the clip should
begin with a neutral-facing section so the existing 45-frame pose calibration
is representative; the saved five-direction gaze profile replaces interactive
live gaze calibration.

## Install

From the repository root in PowerShell, install the same model/runtime extras
used by the live monitor:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[monitor]"
```

Face recognition additionally needs the existing recognition dependency and a
local driver database:

```powershell
python -m pip install -e ".[monitor,recognition]"
```

The evaluator uses the repository's existing `gaze-calibration.json` by default.
Provide another previously saved profile with `--calibration`; it does not
perform interactive camera calibration.

## Ground-truth interval CSV

For `behavior`, `gaze`, `fatigue`, and `recognition`, provide a CSV with these
columns. `start_time` and `end_time` accept seconds (including decimals) or
`HH:MM:SS[.fraction]`; intervals are start-inclusive and end-exclusive. Use
one label per non-overlapping interval. Videos match by basename.

```csv
video_name,start_time,end_time,actual_label
drive01.mp4,0,8,NORMAL
drive01.mp4,8,11,FATIGUE ALERT
```

Run the classifier/state machine evaluations (examples):

```powershell
python -m evaluation.evaluate_video --task behavior `
  --videos ".\data\evaluation\drive01.mp4" `
  --ground-truth ".\data\evaluation\labels.csv" `
  --output-dir ".\evaluation-output\behavior"

python -m evaluation.evaluate_video --task fatigue `
  --videos ".\data\evaluation\drive01.mp4" `
  --ground-truth ".\data\evaluation\fatigue-labels.csv"

python -m evaluation.evaluate_video --task gaze `
  --videos ".\data\evaluation\drive01.mp4" `
  --ground-truth ".\data\evaluation\gaze-labels.csv" `
  --calibration ".\gaze-calibration.json"
```

For the annotations currently stored in this repository, select only videos
named by each CSV (PowerShell):

```powershell
$behaviorVideos = Import-Csv ".\data\evaluator\behavior-labels.csv" |
  Select-Object -ExpandProperty video_name -Unique |
  ForEach-Object { ".\data\evaluator\$_" }

python -m evaluation.evaluate_video --task behavior `
  --videos $behaviorVideos `
  --ground-truth ".\data\evaluator\behavior-labels.csv" `
  --output-dir ".\evaluation-output\behavior"

$gazeVideos = Import-Csv ".\data\evaluator\gaze-labels.csv" |
  Select-Object -ExpandProperty video_name -Unique |
  ForEach-Object { ".\data\evaluator\$_" }

python -m evaluation.evaluate_video --task gaze `
  --videos $gazeVideos `
  --ground-truth ".\data\evaluator\gaze-labels.csv" `
  --calibration ".\gaze-calibration.json" `
  --output-dir ".\evaluation-output\gaze"
```

Gaze ground truth maps `STRAIGHT` and `FORWARD` to `FORWARD`,
`LOOKING_UP` and `UP` to `UP`, and preserves `RIGHT`, `LEFT`, `DOWN`, and
`UNKNOWN`. Directional metrics use the same five labels for actual and
predicted values. `UNKNOWN`, calibration frames reported by the runtime, and
invalid/missing gaze predictions are excluded and counted separately. Unlabeled
time gaps remain unlabeled. The report also compares raw directions with the
live debounced direction and records the `FORWARD`-actual-to-`UP` counts and
gaze-coordinate evidence. It separately reports frame-level non-forward
direction and final sustained-distraction-alert metrics; those alert metrics
are not event-level matches or false-alerts-per-minute measurements.

The standalone SVM emotion model can also be measured without driver-profile
recognition or live-worker scheduling. It uses the existing face landmark
adapter, face-crop helper, and `SVMEmotionAnalyzer.predict` method:

```powershell
python -m evaluation.evaluate_video --task emotion `
  --videos ".\data\evaluation\emotion01.mp4" `
  --ground-truth ".\data\evaluation\emotion-labels.csv"
```

`fatigue` labels may be positive/negative, true/false, yes/no, drowsy/normal,
or alert/clear. Recognition labels are driver IDs; use `unknown` for unknown
drivers:

```powershell
python -m evaluation.evaluate_video --task recognition `
  --videos ".\data\evaluation\known-and-unknown.mp4" `
  --ground-truth ".\data\evaluation\identity-labels.csv" `
  --database ".\data\local\drivers_database.pkl"
```

## Fatigue event evaluation

The existing `fatigue-labels.csv` supports frame-level evaluation only. For
event precision/recall, false-alert rate, missed events, and detection delay,
the event manifest must include ground-truth and prediction event rows, the
matching model hash, and independent test-split metadata. Optional
`clip_duration_seconds` is required on every clip to compute false alerts per
minute.

The frame-level fatigue report compares three stages against the labelled
binary intervals: current eye-state closure, sustained closure at the existing
confirmation duration, and the final fatigue alert. Frames with no valid EAR
or eye state, missing faces, or calibration status are excluded and counted
separately. These frame metrics do not replace temporal event matching.

```powershell
python -m evaluation.evaluate_video --task fatigue `
  --event-manifest ".\data\evaluator\fatigue-events.csv" `
  --model ".\path\to\fatigue-model-file" `
  --split-metadata ".\data\evaluator\split-metadata.json" `
  --output-dir ".\evaluation-output\fatigue"
```

The model file must be the one whose SHA-256 is recorded on every prediction
event row. Event matching uses temporal overlap and one-to-one assignment;
duplicate predictions count as false-positive events. Event precision/recall/F1
and delay are reported separately from consistent clip-level TP/TN/FP/FN and
false-positive/negative rates. No event annotations or split metadata are
currently present in `data\evaluator`, so event results remain pending.

When there is no usable fatigue clip or event dataset, record that explicitly
without retaining stale metrics from an earlier run:

```powershell
python -m evaluation.evaluate_video --task fatigue `
  --ground-truth ".\data\evaluator\fatigue-labels.csv" `
  --pending-reason "The referenced video and event manifest are unavailable." `
  --output-dir ".\evaluation-output\fatigue"
```

This writes a `pending_data` report with null metrics and removes only an
obsolete fatigue confusion-matrix image.

## YOLO bounding-box CSV

YOLO evaluation uses one row per labelled frame/object. A labelled empty frame
must still have a row with blank `class_name` and blank coordinates. Boxes use
pixel-space `xyxy` coordinates. The supported model mapping comes from the
existing detector: `0=Phone`, `1=Cigarette`, `2=Cup`, `3=Bottle`. The evaluator
checks that every decoded frame is annotated, including empty frames, and
rejects invalid/non-finite or out-of-frame coordinates.

```csv
video_name,frame_index,class_name,x_min,y_min,x_max,y_max
drive01.mp4,0,Phone,120,80,240,310
drive01.mp4,1,,,,,
```

```powershell
python -m evaluation.evaluate_video --task yolo `
  --videos ".\data\evaluation\drive01.mp4" `
  --annotations ".\data\evaluation\boxes.csv" `
  --output-dir ".\evaluation-output\yolo"
```

The YOLO report includes precision/recall/F1 at IoU 0.50, mean matched IoU,
AP50, AP50:95, per-class results, and a background-inclusive confusion matrix.
Raw boxes and live class-confidence-filtered boxes are scored separately.
Phone persistence/proximity and final phone-use alerts are not scored without
independent driver identity/face-location annotations. Object detections are
not treated as proof of driver distraction.

If labels are not ready, create an explicit pending report without scoring:

```powershell
python -m evaluation.evaluate_video --task yolo `
  --videos $behaviorVideos `
  --output-dir ".\evaluation-output\yolo"
```

The checked-in `data/evaluator/boxes.csv` is not a usable dataset: it contains
only a few frame rows, symbolic coordinate placeholders such as `LEFT`/`TOP`,
and refers to `driver1.mp4`, which is not among the checked-in evaluator videos.
Do not pass it as valid ground truth; prepare complete numeric annotations
before requesting YOLO metrics.

### Converting CVAT video-track exports

No CVAT XML export is currently present in the repository. To avoid treating
unreviewed frames as negative examples, prepare a coverage CSV that explicitly
lists every frame reviewed by the annotator. Use `annotated` for frames with
one or more configured object boxes and `verified_empty` only for frames
reviewed and confirmed to contain none of the four configured objects:

```csv
video_name,frame_index,status
drive1.avi,0,annotated
drive1.avi,1,verified_empty
```

Convert a CVAT video-track XML export with:

```powershell
python -m evaluation.cvat_to_yolo `
  --xml ".\data\evaluator\cvat-export.xml" `
  --coverage ".\data\evaluator\cvat-coverage.csv" `
  --output ".\data\evaluator\boxes-from-cvat.csv"
```

The converter maps CVAT `track` labels and `box` attributes
(`frame`, `xtl`, `ytl`, `xbr`, `ybr`) directly to evaluator pixel-space xyxy
rows. CVAT `outside="1"` boxes are omitted. Only the exact configured class
names `Phone`, `Cigarette`, `Cup`, and `Bottle` are accepted. Coverage frames
without boxes become blank-class rows only when explicitly marked
`verified_empty`; absent coverage entries remain unannotated, and the YOLO
evaluator rejects incomplete coverage against the selected video's decoded
frames. Review the XML and coverage file together before scoring.

## Outputs and interpretation

Every run writes:

* `predictions.csv` with frame index, video-time timestamp, actual label,
  prediction, and applicable scores/details.
* `report.json` and `report.csv` with classification/detection metrics, rates,
  and confusion counts.
* `confusion_matrix.png` when the task has a confusion matrix. Pending YOLO
  reports contain no fabricated metrics or confusion image.

Classification reports include accuracy, per-class precision/recall/F1,
macro and weighted F1, one-vs-rest false-positive/negative rates, and a
confusion matrix. Recognition
reports separately include false-accept rate (unknown labelled faces classified
as known) and false-reject rate (known labelled faces classified as unknown).
This is an offline measurement, not a change to live monitoring or calibration.
Offline behavior evaluation runs classifier inference with one job to avoid
all-core parallel overhead on each video frame.

### Behavior evaluation protocol

Behavior evaluation scores only the four ground-truth behavior classes:
`NORMAL`, `YAWNING`, `DISTRACTED`, and `TALKING`. Ground-truth labels are
case-normalized and matched to each decoded frame by exact video basename and
the half-open interval `[start_time, end_time)`. Frames outside all labelled
intervals remain in `predictions.csv` as `UNLABELLED` but are not used in
classification metrics. Labelled frames whose final prediction is
`CALIBRATING` are excluded from behavior metrics and counted separately.

The scored prediction is `predicted_label`, the post-temporal-filter behavior
output consumed by the monitor's dashboard and alerts. `raw_model_behavior`
records the direct classifier output, while `raw_behavior` records the
pre-temporal decision after existing overrides. The report provides separate
raw-model and final-decision metrics. The confusion matrix contains only the
four supported behavior classes. Face-not-detected and unclassified outputs
are excluded from it and reported separately; missing faces are never counted
as `NORMAL`.

The report separately counts actual face-detection misses using
`face_present=False` on labelled frames. Such intervals are treated as
face-expected because the interval CSV has no independent face-visibility
annotation. The count of `FACE NOT DETECTED` output labels is also reported;
it can exceed actual face misses when the temporal behavior output remains in
that operational state after face reacquisition. The confusion matrix has four
ground-truth rows and four behavior plus operational prediction columns.
False-positive and false-negative rates are reported per behavior class and as
an unweighted macro average over the same four ground-truth classes.

For behavior runs, `report.json` includes total decoded frames, labelled and
unlabelled frames, calibration exclusions, evaluated frames, face-detection
metrics, and per-video checks for filename matching, decoded frame count,
timestamps, annotation labels, and annotation bounds against video duration.
`predictions.csv` includes `evaluation_status` values such as `UNLABELLED`,
`EXCLUDED_CALIBRATING`, `EXCLUDED_FACE_NOT_DETECTED`,
`EXCLUDED_UNCLASSIFIED`, and `EVALUATED` to make row selection auditable.

## Threshold audit and combined summary

The offline threshold audit reads the saved gaze profile, records current
integrated production defaults and source symbols, and includes raw/final
performance from reports already present under the output root. It does not
tune or change thresholds:

```powershell
python -m evaluation.evaluate_video --task threshold-audit `
  --calibration ".\gaze-calibration.json" `
  --output-dir ".\evaluation-output"
```

To produce the combined report after running module evaluations:

```powershell
python -m evaluation.evaluate_video --task summary `
  --output-dir ".\evaluation-output"
```

This writes `combined-summary.json` and `combined-summary.csv`. It lists each
module's status and metrics without combining behavior/gaze frame accuracy,
fatigue event/clip metrics, or YOLO box metrics into one score.

The audited integrated defaults are documented in the source files referenced
by `threshold-audit.json`. In particular: eye closure/opening EAR thresholds
are `0.21`/`0.24` with two confirming frames; fatigue confirms after `1.5 s`
closed and clears after `0.35 s` open; MAR yawn activation/release is `0.62`/
`0.52` with 12 frames; talking uses 24 MAR samples, mean delta `0.035`, and at
least two crossings. Head-pose limits are 25 degrees yaw and 20 degrees pitch,
with 18 turned frames and 8 forward-recovery frames. The pose calibrator uses
45 frames and smoothing alpha `0.35`. Behavior persistence is 0.35 s talking,
0.15 s yawning, 0.10 s distraction, and 0.15 s clear-to-normal.

Gaze uses the saved personalized boundaries, a 0.05 horizontal fallback,
5-frame voting with 4 votes required, and 0.6 s integrated non-forward
persistence. YOLO uses 416-pixel inference, base confidence `0.10`, NMS IoU
`0.45`, max 20 detections, and per-class confidence `Phone=0.20`,
`Cigarette=0.75`, `Cup=0.35`, `Bottle=0.40`. Phone-use alerts additionally
require 0.30 s track confirmation, confirmed-driver proximity, 0.75 s use
persistence, and 0.50 s alert persistence. Values are audited, not optimized.

## Current dataset limitations

The behavior annotation CSV covers the 12 checked-in `.avi` clips. Gaze labels
cover six of those clips and currently include `FORWARD`, `UNKNOWN`, and other
direction labels (the normalizer also supports the requested `STRAIGHT` and
`LOOKING_UP` spellings). The available fatigue interval CSV contains one
`normal` interval for `driver1.mp4`, but that video is absent; the available
event manifest/split metadata required for event metrics are also absent.
YOLO annotations are incomplete and malformed as described above. These
limitations must remain visible as pending/unavailable evaluations, not
replaced with generated labels or synthetic project scores.

Run the framework's metric and input tests from the repository root:

```powershell
python -m unittest tests.unit.test_offline_evaluation_metrics tests.unit.test_evaluation_framework -v
```
