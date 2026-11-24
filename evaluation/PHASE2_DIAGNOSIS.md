# Phase 2 Offline Evaluation Diagnosis

## Executive summary

- The behavior pickle records scikit-learn 1.7.2; the project environment was
  pinned to 1.7.1. A separate Python 3.11 evaluation environment now uses
  scikit-learn 1.7.2. Loading the pickle there produced zero
  `InconsistentVersionWarning` warnings.
- The project pin, global Python environment, live monitor, model files, and
  production thresholds were not changed. The previous reports remain intact.
- Compatible behavior and gaze reruns are in
  `../evaluation-output/phase2-diagnosis/baseline-sklearn-1.7.2/`.
- The behavior results show an association between post-rule/temporal decisions
  and lower macro F1, especially for TALKING. Per-filter causal attribution is
  unavailable because prediction records do not include filter-reason traces.
- Gaze label normalization is verified. Most actual-FORWARD-to-UP errors have
  gaze-y values below the existing personalized UP boundary; this is evidence
  for further coordinate/calibration review, not justification to change a
  production threshold.
- Fatigue event evaluation and YOLO box scoring remain pending valid inputs.
  No independent validation split was found, so no threshold sweep or
  recommended threshold is supported.

## Compatibility resolution

The original behavior model pickle at
`models/behavior/driver_behavior_model.pkl` was serialized with scikit-learn
1.7.2. The normal project runtime and `pyproject.toml` pin were 1.7.1, so the
old behavior scores were provisional.

An isolated environment was created at
`%LOCALAPPDATA%\Driver-Anomly-eval-py311` using Python 3.11 and
`--system-site-packages`; only scikit-learn 1.7.2 was installed into that
environment. The pickle loaded as a `RandomForestClassifier` with zero
inconsistent-version warnings. The 1.7.1 project pin was intentionally
preserved. Compatible reruns were written to a new directory and did not
overwrite the prior reports.

## Compatible baseline results

| Module | Compatible result | Data status |
|---|---|---|
| Behavior | Accuracy 0.5484; macro F1 0.4253; weighted F1 0.4929 on 22,194 scored frames | 11 of 12 annotated videos included |
| Gaze | Smoothed-direction accuracy 0.4794; macro F1 0.3355 on 9,913 directional frames | 6 annotated videos |
| Fatigue | Pending; no event manifest, independent split metadata, or referenced video | No event metrics |
| YOLO | Pending; no complete numeric bounding-box annotations or CVAT export | No detection metrics |

The prior 1.7.1 baseline report directories remain the original baseline.
Compatible report files:

- Behavior: `../evaluation-output/phase2-diagnosis/baseline-sklearn-1.7.2/behavior/`
- Gaze: `../evaluation-output/phase2-diagnosis/baseline-sklearn-1.7.2/gaze/`

Each evaluated directory contains `report.json`, `report.csv`,
`predictions.csv`, and `confusion_matrix.png`. The earlier
`../evaluation-output/combined-summary.json` and all original module reports
were not regenerated.

## Behavior diagnosis

The compatible run processed 24,762 frames; 23,381 were labelled and 22,194
were eligible for the four-class final confusion matrix. Exclusions were 1,381
unlabelled frames, 556 calibration frames, 61 face-missing frames, and 570
unclassified outputs. Face-missing frames are reported separately and are not
counted as NORMAL. The `driver10.avi` annotation extends to 42 seconds, beyond
the 40.566-second video duration, so that video was excluded rather than
silently truncating its labels.

To compare decision stages fairly, the following metrics were recomputed from
the same 22,194 final-valid frames:

| Stage | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Direct classifier (`raw_model_behavior`) | 0.5075 | 0.4952 | 0.5207 |
| Rule-adjusted, pre-temporal (`raw_behavior`) | 0.5550 | 0.4497 | 0.5109 |
| Final filtered (`predicted_label`) | 0.5484 | 0.4253 | 0.4929 |

The direct-to-pre-temporal stage includes existing decision-rule overrides as
well as the classifier output. The pre-temporal-to-final comparison isolates
the behavior temporal-filter stage on this cohort. Its accuracy decreases by
0.0066 and macro F1 by 0.0244. The F1 changes by class are:

| Class | Pre-temporal F1 | Final F1 | Final precision | Final recall |
|---|---:|---:|---:|---:|
| NORMAL | 0.6910 | 0.6881 | 0.5814 | 0.8427 |
| YAWNING | 0.6908 | 0.6607 | 0.8827 | 0.5280 |
| DISTRACTED | 0.2934 | 0.2751 | 0.2892 | 0.2622 |
| TALKING | 0.1237 | 0.0773 | 0.2045 | 0.0476 |

There were 2,128 pre-temporal-to-final label changes in the common cohort. The
most frequent were TALKING to NORMAL (854), NORMAL to DISTRACTED (322), NORMAL
to TALKING (260), DISTRACTED to NORMAL (204), and YAWNING to NORMAL (157).
These changes, along with the class-wise decline, associate the final-stage
drop with temporal filtering and a large shift toward NORMAL. The records do
not identify which internal filter caused each change; no causal claim or
threshold recommendation is made.

The final confusion matrix and class distributions are in the compatible
behavior report. TALKING is the clearest weakness: only 236 of 4,954 TALKING
frames were correctly retained in the final class, and recall is 0.0476.
Review annotation timing and representative transitions before drawing a
model-quality conclusion.

## Gaze diagnosis

The normalized ground-truth labels and predicted class set are consistent:
`STRAIGHT` maps to `FORWARD`, `LOOKING_UP` maps to `UP`, and `RIGHT`, `LEFT`,
and `UNKNOWN` retain their meaning. `UNKNOWN` intervals, calibration frames,
unlabelled frames, and invalid predictions are excluded and reported
separately. The compatible run processed 11,684 frames: 968 unlabelled, 270
calibration, 540 UNKNOWN-ground-truth, and 7 invalid-prediction frames were
excluded, leaving 9,913 directional samples.

| Direction | Actual | Predicted (smoothed) | False positives | False negatives | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FORWARD | 5,986 | 3,698 | 883 | 3,171 | 0.7612 | 0.4703 | 0.5814 |
| RIGHT | 1,276 | 1,105 | 526 | 697 | 0.5240 | 0.4538 | 0.4864 |
| LEFT | 2,621 | 1,876 | 548 | 1,293 | 0.7079 | 0.5067 | 0.5906 |
| UP | 30 | 3,114 | 3,084 | 0 | 0.0096 | 1.0000 | 0.0191 |
| DOWN | 0 | 120 | 120 | 0 | 0.0000 | 0.0000 | 0.0000 |

DOWN has no ground-truth support, so its recall and F1 are not evidence of
performance on DOWN examples. The confusion matrix is preserved in the gaze
report.

| Gaze/decision stage | Accuracy | Macro F1 | Notes |
|---|---:|---:|---|
| Raw direction (`raw_gaze_label`) | 0.4935 | 0.3507 | Before gaze voting/debounce |
| Smoothed direction (`predicted_label`) | 0.4794 | 0.3355 | Existing production smoothing |
| Smoothed non-FORWARD decision | 0.5910 | 0.5908 | Frame-level binary comparison |
| Final sustained distraction alert | 0.6147 | 0.6040 | Frame-level only; positive-class precision 0.5124, recall 0.5684, F1 0.5389 |

The final distraction-alert numbers are not event-level alert matches. The
current annotations do not support event delay, missed-event duration, or
false alerts per minute.

The existing personalized calibration has an UP boundary of
`gaze_y < -0.073257` (`up_is_lower=true`). Among actual-FORWARD frames predicted
UP, the mean/median gaze-y values were -0.078966/-0.079305; for actual-FORWARD
frames predicted FORWARD they were -0.053443/-0.057664. There were 2,395 raw UP
candidates and 2,448 smoothed UP predictions among actual-FORWARD frames; 335
raw FORWARD frames changed to smoothed UP, while 273 changed from UP to
FORWARD. These values show that many errors cross the existing UP boundary.
They do not determine whether calibration, coordinate distributions, video
pose, or annotation alignment is the cause. Do not change calibration or
thresholds from this evidence alone.

## Fatigue evaluation status and required inputs

Fatigue event evaluation remains pending. The only current interval annotation
is `driver1.mp4,0,56.48,normal`; that video is absent. There is no event
manifest or independent split metadata. Required event-manifest columns are:

```text
clip_id,identity_id,sha256,model_sha256,source,start_seconds,end_seconds
```

`source` must be `ground_truth`, `prediction`, or `negative_clip`.
`model_sha256` must be populated only for prediction rows and match the
provided model file. Add `clip_duration_seconds` for every clip to compute
false alerts per minute. Include a clip's events only with valid finite
timestamps and a duration that bounds them.

The split metadata must declare `independent_test: true`, include
`training_identity_ids`, `test_identity_ids`, `training_sha256`, and
`test_sha256`, and have no identity or file-hash overlap. Every manifest
identity and sample hash must belong to the test sets. Until these files and
the referenced clips exist, TP/FP/TN/FN, delay, event precision/recall, and
false-alert rates are not reported.

## YOLO evaluation status and CVAT conversion

No CVAT XML export exists in the repository. The checked-in
`data/evaluator/boxes.csv` has placeholder coordinates (`LEFT`, `TOP`, etc.),
only a few frame rows, and names `driver1.mp4`, which is not an available
evaluator video. It cannot provide a valid detection score. The configured
classes are IDs `0=Phone`, `1=Cigarette`, `2=Cup`, and `3=Bottle`; no other
class IDs are assumed.

The evaluator requires one or more numeric pixel-space xyxy boxes for each
positive frame and a blank-class row for every verified-empty frame. Missing
frame entries mean unannotated, not negative. The new
`evaluation/cvat_to_yolo.py` converter accepts CVAT video-track XML plus an
explicit coverage CSV:

```csv
video_name,frame_index,status
driver1.avi,0,annotated
driver1.avi,1,verified_empty
```

Use `annotated` only when the frame has one or more configured object boxes;
use `verified_empty` only after review confirms no configured objects. The
converter maps CVAT `frame`, `xtl`, `ytl`, `xbr`, `ybr`, and `label`, and omits
boxes with `outside="1"`. It rejects boxes outside the declared coverage and
never treats missing coverage as a negative. The evaluator then checks coverage
against all decoded frames of the selected video.

```powershell
python -m evaluation.cvat_to_yolo `
  --xml ".\data\evaluator\cvat-export.xml" `
  --coverage ".\data\evaluator\cvat-coverage.csv" `
  --output ".\data\evaluator\boxes-from-cvat.csv"
```

The converter is covered by synthetic unit tests only; it has not been run on
project CVAT annotations because none were found. No YOLO precision, recall,
F1, IoU, AP50, AP50:95, or mAP score is claimed. Object detections are not
evidence of driver distraction, and the final proximity/persistence alert is
not measurable without independent driver/face and alert-event annotations.

## Production threshold audit and sensitivity

The static threshold audit is preserved at
`../evaluation-output/threshold-audit.json`. The values below are read from
the existing source/configuration; none were changed. The report's performance
fields are the old baseline snapshot; use the compatible reports above for
Phase 2 scores.

| Module | Existing decision values | Source files |
|---|---|---|
| Behavior | Confidence 0.65; yaw 25 degrees; pitch 20 degrees; EAR close/open 0.21/0.24 with two confirmation frames; fatigue confirm 1.5 s, clear 0.35 s; yawn MAR 0.62, release 0.52, 12 frames, 4 s rearm; talking window 24 frames, mean MAR delta 0.035, 2 crossings; head distraction 18 consecutive frames, forward recovery 8; temporal persistence talk/yawn/distraction/clear = 0.35/0.15/0.10/0.15 s | `src/driver_monitor/monitoring/legacy_runtime.py`, `src/driver_monitor/monitoring/temporal.py` |
| Gaze | Personalized left/right 0.389715/0.601829; up/down -0.073257/-0.005108; debounce 5 frames / 4 votes; distraction persistence 0.6 s; calibration capture 1.5 s, minimum 8 samples | `Phase-3/Buffered-system/buffered_system/Buffer/Buffered_system/gaze_detection.py`, `src/driver_monitor/monitoring/legacy_runtime.py` |
| Fatigue | EAR close/open 0.21/0.24; close/open confirmation 2 frames; fatigue confirmation 1.5 s; recovery open 0.35 s; blink-rate changes do not trigger alerts | `Phase-3/Buffered-system/buffered_system/Buffer/Buffered_system/train_and_run_driver_detector.py`, `src/driver_monitor/monitoring/legacy_runtime.py` |
| YOLO | Image size 416; base confidence 0.10; NMS IoU 0.45; maximum 20 detections; class confidence Phone/Cigarette/Cup/Bottle = 0.20/0.75/0.35/0.40; max box area 0.85; phone confirmation 0.30 s / two hits; driver-use persistence 0.75 s and alert persistence 0.50 s | `src/driver_monitor/monitoring/legacy_runtime.py`, `src/driver_monitor/detection/objects.py`, `src/driver_monitor/monitoring/temporal.py` |

No independent validation split is available for behavior, gaze, or YOLO
threshold analysis. A sweep on the current held-out evaluation annotations
would select values on the same data used for final performance reporting, so
no experimental threshold sweep was run and no recommended threshold is
provided. If tuning is later requested, use a separate validation split, freeze
the selected values, and report final metrics once on an untouched independent
test split. Production values must remain separate from any offline experiment.

## Follow-up experiments

1. Correct or explicitly exclude the out-of-bounds `driver10.avi` behavior
   annotation, then rerun the full behavior video set.
2. Review representative TALKING-to-NORMAL and other frequent transitions
   against source frames and annotation timing. Add evaluation-only filter
   decision traces if the runtime exposes them; do not infer a specific cause
   from transition counts alone.
3. Break down actual-FORWARD gaze-y distributions per video and inspect
   calibration segments and labels near the existing UP boundary. Keep
   production calibration and thresholds fixed during diagnosis.
4. Export CVAT video-track XML and a reviewed per-frame coverage CSV for all
   selected YOLO videos. Include every decoded frame, including verified-empty
   frames, before calculating detection metrics.
5. Provide the fatigue event manifest, verified negative clips, matching
   model hash, independent split metadata, and the referenced video files.
6. Only after an independent validation set exists, run an explicitly
   experimental sensitivity analysis; do not tune on the final test set.

## Validation and scope

- Model compatibility: verified under Python 3.11 / scikit-learn 1.7.2;
  zero `InconsistentVersionWarning` warnings on pickle load.
- Compatible behavior and gaze baseline reruns completed in the separate
  Phase 2 output directory.
- CVAT conversion tests use tiny synthetic fixtures; no synthetic data was
  used in project evaluation reports.
- Fatigue and YOLO remain pending for missing ground truth rather than
  receiving fabricated metrics.
- The project dependency pin, live monitoring code, model weights, and
  production thresholds were not modified.
