# YOLO False-Positive Diagnosis

## Scope and conclusion

This is a read-only diagnosis of the current application and checked-in evidence.
No live-monitoring code, thresholds, model weights, camera/worker behavior,
buffering, alerts, or dashboard behavior were changed. The only deliverable is
this report.

This report records the pre-implementation state inspected during the
investigation. The subsequent temporal-validation implementation is documented
in `docs/PROJECT_ARCHITECTURE_AND_METRICS.md`.

The strongest confirmed cause of immediate object false alerts is in software:
the shared object-history filter requires only one class occurrence
(`OBJECT_DETECTION_MIN_HITS = 1`), and the class-presence check does not track
object identity. Consequently, a single class-confidence-passing Cigarette,
Cup, or Bottle box can reach the object-alert and dashboard paths; Cigarette
and Cup also trigger object-specific audio, while Bottle does not. Phone boxes
also display immediately, but the phone-use alert itself has separate
track/persistence/driver-proximity checks. A dashboard fallback bypasses those
checks when it renders a raw phone detection as an alert status.

The model may also produce class-confused boxes, but the repository does not
contain the training dataset, its YAML, training/validation annotations, split
manifest, or a usable YOLO test set. Therefore the source of the model's
false-positive predictions and its class-wise quality cannot be confirmed
from current data. The embedded checkpoint metrics are aggregate metadata, not
an independently reproducible evaluation.

## Complete execution path

1. **Entry and setup.** `python -m driver_monitor` enters through
   `src/driver_monitor/__main__.py` and `src/driver_monitor/app.py`; it loads
   `src/driver_monitor/monitoring/legacy_runtime.py`. `ProjectPaths` resolves
   the active weights to `models/objects/best_phone_finetuned_v1.pt`.
   `BufferedDriverMonitor.calibrate_gaze()` starts the camera, Telegram worker,
   and object worker before gaze calibration completes. `start()` then starts
   the remaining monitor workers and audio worker.
2. **Capture and scheduling.** `CameraProducer.run()` in
   `src/driver_monitor/monitoring/camera.py` reads OpenCV frames and publishes
   `FramePacket`s to the one-slot `LatestFrameBuffer`. New packets replace
   stale ones. `ObjectDetectionWorker.run()` in
   `src/driver_monitor/monitoring/legacy_runtime.py` reads the newest packet
   and runs YOLO on every other captured frame
   (`OBJECT_DETECTION_FRAME_SKIP = 2`).
3. **Inference and post-processing.** `YoloObjectDetector.detect()` in
   `src/driver_monitor/detection/objects.py` calls Ultralytics
   `model.predict()` at image size 416, base confidence 0.10, NMS IoU 0.45,
   and maximum 20 boxes. It maps class IDs, applies the per-class confidence
   cutoffs, converts boxes to integer `xyxy`, and rejects boxes larger than
   85% of the frame in the live worker. `YoloObjectDetector.__init__()` calls
   `_validate_class_names()` and fails if the model's first four names do not
   match the expected class names.
4. **History and phone validation.** The worker appends a set of detected class
   IDs to a three-inference history, counts class occurrences, and retains
   current detections whose class count is at least one. It then sends the
   remaining detections to `PhoneDetectionValidator.update()`. Only class 0
   (Phone) is tracked by this validator. The worker publishes the detections
   and phone state to `ResultBuffer`.
5. **Dashboard, alerts, audio, and Telegram.** `BufferedDriverMonitor.display()`
   snapshots shared results, calls `AlertManager.update()`, calls
   `AlertAudioManager.update()`, and then renders the dashboard. The renderer
   draws the published detection boxes. `AlertManager` schedules eligible
   alerts and submits non-bottle notifications to `TelegramWorker`; the worker
   queues a snapshot and later invokes the existing Telegram integration.
   `AlertAudioManager` queues a one-shot beep and voice message on newly active
   object-class conditions. Bottle is local-only and has no object-specific
   audio message.

The packaged desktop application is the active path. The older standalone
`Phase-2/Object-Detection/webcam.py` is a separate webcam path: it repeats
inference and the same one-hit class-history pattern, but checks only that
model class IDs exist (not that their names match) and uses a Cigarette
confidence cutoff of 0.50 rather than the packaged runtime's 0.75. Its Phone,
Cup, and Bottle cutoffs match the packaged runtime; do not treat it as the
packaged application's active worker.

## Current live thresholds and conditions

| Setting | Current value | Source |
|---|---:|---|
| Inference image size | 416 | `OBJECT_DETECTION_IMAGE_SIZE` in `src/driver_monitor/monitoring/legacy_runtime.py` |
| Base confidence | 0.10 | `OBJECT_DETECTION_BASE_CONFIDENCE` in `src/driver_monitor/monitoring/legacy_runtime.py` |
| NMS IoU | 0.45 | `OBJECT_DETECTION_IOU_THRESHOLD` in `src/driver_monitor/monitoring/legacy_runtime.py` |
| Maximum boxes | 20 | `OBJECT_DETECTION_MAX_DETECTIONS` in `src/driver_monitor/monitoring/legacy_runtime.py` |
| Inference cadence | Every 2nd captured frame | `OBJECT_DETECTION_FRAME_SKIP` in `src/driver_monitor/monitoring/legacy_runtime.py` |
| Class-history length | 3 inference results | `OBJECT_DETECTION_HISTORY_LENGTH` in `src/driver_monitor/monitoring/legacy_runtime.py` |
| Class-history minimum | 1 occurrence | `OBJECT_DETECTION_MIN_HITS` in `src/driver_monitor/monitoring/legacy_runtime.py` |
| Phone confidence | 0.20 | `OBJECT_DETECTION_CLASS_CONFIDENCE` in `src/driver_monitor/monitoring/legacy_runtime.py` |
| Cigarette confidence | 0.75 | `OBJECT_DETECTION_CLASS_CONFIDENCE` in `src/driver_monitor/monitoring/legacy_runtime.py` |
| Cup confidence | 0.35 | `OBJECT_DETECTION_CLASS_CONFIDENCE` in `src/driver_monitor/monitoring/legacy_runtime.py` |
| Bottle confidence | 0.40 | `OBJECT_DETECTION_CLASS_CONFIDENCE` in `src/driver_monitor/monitoring/legacy_runtime.py` |
| Maximum box area | 0.85 of frame | `ObjectDetectionWorker.run()` passes `max_box_fraction=0.85` to `YoloObjectDetector.detect()` |
| Phone track matching | IoU > 0.10 | `PHONE_TRACK_IOU_THRESHOLD` in `src/driver_monitor/monitoring/temporal.py` |
| Phone presence confirmation | At least 2 hits and 0.30 s persistence | `PhoneDetectionValidator.__init__()` / `update()` in `src/driver_monitor/monitoring/temporal.py`; wired by `ObjectDetectionWorker` |
| Driver phone-use suspicion | 0.75 s near a confirmed driver | `--driver-phone-suspicion-seconds` default in `legacy_runtime.main()` and `PhoneDetectionValidator.update()` |
| Phone alert persistence | 0.50 s | `--driver-phone-alert-persistence-seconds` default in `legacy_runtime.main()` and `PhoneDetectionValidator.update()` |
| Phone track retention | 2 missed processed frames; stale after 1.0 s | `ObjectDetectionWorker` default and `PhoneDetectionValidator.__init__()` in `src/driver_monitor/monitoring/temporal.py` |
| Telegram cooldown | 60 s per alert key | `legacy_runtime.main()` and `AlertManager.__init__()` in `src/driver_monitor/monitoring/alerts.py` |
| Audio object event | One beep/voice on newly active class | `AlertAudioManager.update()` in `src/driver_monitor/monitoring/sound.py` |

These are observed production defaults, not recommendations to change them.

## Per-class false-positive behavior

The active IDs are Phone=0, Cigarette=1, Cup=2, Bottle=3. The checkpoint's
embedded `names` map is lowercase with the same ID order; runtime validation
compares names case-insensitively.

| Class | Actual alert condition in the packaged runtime | Dashboard/audio/Telegram result |
|---|---|---|
| **Phone** | Raw box passes confidence 0.20 to display immediately. Safety alert requires `driver_phone_alert`, `driver_confirmed`, and `phone_detection_confirmed`; the temporal validator requires at least two matching detections over 0.30 s, driver proximity while the driver is confirmed, 0.75 s use persistence, then 0.50 s alert persistence. | Raw Phone box and confidence are drawn immediately. `dashboard_alert_status()` falls back to the raw detection class if `alert_status == "NONE"` and can show “POSSIBLE DRIVER PHONE USE” without the phone-use validation. In contrast, `AlertManager` and audio require the validated driver-phone condition. Eligible phone alerts can be sent to Telegram and audio. |
| **Cigarette** | Any current Cigarette detection that passes 0.75 is enough; `AlertManager.update()` reads raw published `object_detections` (non-phone IDs), with no Cigarette track or persistence check. The first active-class transition becomes pending and is eligible for notification subject to priority/cooldown. | Detection box is drawn; raw class can appear as dashboard alert status. `AlertAudioManager.update()` triggers a beep/voice on the class's newly active transition. Eligible Cigarette alert is submitted to Telegram. |
| **Cup** | Any current Cup detection that passes 0.35 is enough; same raw-detection path as Cigarette, with no Cup track or persistence check. | Detection box/dashboard status; one-shot beep/voice on activation; eligible Cup alert is submitted to Telegram. |
| **Bottle** | Any current Bottle detection that passes 0.40 is enough; same raw-detection path, with no Bottle track or persistence check. | Detection box/dashboard status; `AlertManager` marks Bottle as local-only instead of submitting Telegram. `sound.OBJECT_ALERTS` has no Bottle entry, so no Bottle-specific beep/voice is queued. |

For Cigarette/Cup/Bottle, the nominal history does not create even a
class-level temporal gate: the worker records IDs in sets and uses
`hit_counts[class_id] >= OBJECT_DETECTION_MIN_HITS`, currently `>= 1`.
It does not compare bounding boxes or associate an object across frames.
Changing that minimum alone in a future implementation would still count
class occurrences, not confirm the same object.

`AlertManager.update()` makes object alert entries on a transition from absent
to present; one transient false box is therefore enough to enqueue an alert.
The configured cooldown can defer delivery but is not a detection validator.
`AlertAudioManager.update()` similarly detects a new active class and queues
audio without object-level temporal validation.

## Model, data, and evaluation evidence

### What the checkpoint establishes

The canonical packaged weights are
`models/objects/best_phone_finetuned_v1.pt`. Static inspection of its PyTorch
ZIP/pickle opcodes (without deserializing or executing the checkpoint) found:

- Ultralytics version `8.4.178`.
- Model metadata class names: `0=phone`, `1=cigarette`, `2=cup`, `3=bottle`;
  this agrees with the active runtime's case-insensitive mapping check.
- A source checkpoint path of
  `/kaggle/input/datasets/itsme070/best-model/best_candidate.pt`.
- Fine-tuning data-config path `/kaggle/working/data_finetune.yaml`,
  run name `phone_finetune_v1-2`, `epochs=20`, `pretrained=True`, and
  `cls_remap=True`.
- Stored aggregate `train_metrics`: precision 0.81393, recall 0.72427,
  mAP50 0.82424, and mAP50-95 0.64374. These values have no accompanying
  per-class breakdown, split manifest, class support, or independently
  checkable predictions in the repository.

This names a source model and a training config path, **not the identity or
contents of the fine-tuning dataset**. The referenced YAML is not present in
the repository. No YOLO training script, dataset config, training log,
annotation set, or split manifest was found. Other files named
`train_and_run_driver_detector.py` train facial-behavior systems, not this
YOLO artifact.

The three copies at `models/objects/`,
`Phase-2/Object-Detection/`, and
`Phase-3/Buffered-system/buffered_system/Buffer/Buffered_system/` have the
same SHA-256:
`4755731CB269C7FD8890DA5D70380D63B1849E3DACED7B603AECD3D4DB2777C6`.
This confirms the copies inspected are identical; it does not establish their
training provenance. The binary checkpoint was not loaded or modified.

### Dataset-quality questions and evidence limits

| Question | Evidence-based finding |
|---|---|
| Which dataset trained the fine-tune? | Not identifiable. Metadata only points to `/kaggle/working/data_finetune.yaml`; the file and data it refers to are absent. The Kaggle `itsme070/best-model` path identifies the starting checkpoint, not necessarily the fine-tuning dataset. |
| How many train/validation images per class? | Unknown. There is no available YOLO train/val split, dataset YAML, image-list manifest, or class-count report. Do not infer these counts from filenames or the model metrics. |
| Is there class imbalance? | Cannot be determined without per-class image/instance counts. |
| Are bounding-box annotations accurate? | Cannot be determined; the training annotation files are unavailable. |
| Are hard-negative/background images available? | No training-set evidence is present to confirm or deny this. The 28 loose PNGs in `data/t_data/object/` are not a documented split or a labeled hard-negative set. |
| Is validation independent? | Unknown. No split identity, file-hash, or provenance metadata establishes independence from training. The aggregate checkpoint metrics alone do not prove an independent validation set. |
| Do training and inference class mappings match? | The checkpoint's stored class-name order matches the active runtime mapping by case-insensitive comparison. The actual unavailable training YAML's `names`/`nc` mapping and annotation ID convention cannot be cross-checked. |
| Is there sufficient lighting, angle, occlusion, or object-size variation? | Unknown; no representative training images/split documentation is available. The packaged runtime's 416-pixel inference makes tiny objects a plausible risk, but not a proven cause. |

The loose `data/t_data/object/` directory contains 28 PNGs with mixed
descriptive and generic filenames; it has no visible train/validation
structure or paired YOLO labels. It cannot be treated as the model's training
corpus or used to estimate class distribution or annotation quality.

The repository's own YOLO evaluation is pending rather than scored:
`evaluation-output/yolo/report.json` reports `pending_annotations`,
zero processed/evaluated frames, null metrics, and no prediction rows. The
available `data/evaluator/boxes.csv` contains placeholder coordinates
(`LEFT`, `TOP`, `RIGHT`, `BOTTOM`), only a few rows, and refers to
`driver1.mp4`, which is not an available evaluator clip. It is not valid
ground truth. Thus no independent precision, recall, false-positive rate,
per-class AP, or live-camera false-alert rate is available.

## Confirmed weaknesses and potential causes

### Confirmed software-level weaknesses

1. **No effective persistence for non-phone objects.** Three-frame history
   and `MIN_HITS=1` pass a single qualifying detection. Cigarette, Cup, and
   Bottle have no object-identity tracker or temporal confirmation.
2. **Raw detections drive their alerts.** The published
   `object_detections` are the same confidence-filtered raw detections after
   the no-op-at-one history check. `AlertManager` and `AlertAudioManager` use
   those detections for IDs 1–3.
3. **Dashboard phone status is less strict than phone safety alerts.**
   `dashboard_alert_status()` substitutes a status from raw detections when
   the alert status is `NONE`, including raw Phone class 0. It can therefore
   display “POSSIBLE DRIVER PHONE USE” before phone confirmation, driver
   association, or use persistence. This is inconsistent with the stricter
   `AlertManager` and audio conditions.
4. **Confidence is a filter, not temporal or semantic verification.**
   Per-class thresholds and the model's confidence do not establish that the
   predicted class is correct. The low live cutoffs for Phone (0.20), Cup
   (0.35), and Bottle (0.40) allow more candidate boxes than a stricter
   cutoff would; this is a code fact, not evidence that any particular false
   box has a given score or that thresholds should change.
5. **Validation evidence is not adequate for tuning.** The YOLO evaluation
   report is pending and the only checked-in box CSV is unusable, so neither
   the model nor any alternative threshold can be compared on valid
   independent data.

### Potential model/data weaknesses (not established)

- Unknown dataset provenance, annotation consistency, class balance,
  negative coverage, and split independence prevent ruling out class
  confusion or annotation/split leakage.
- The stored aggregate metrics conceal per-class precision/recall. Overall
  scores can hide a weak class, especially for visually small or similar
  objects.
- Small Cigarette instances, occlusion, motion blur, unusual viewpoints,
  changing cabin lighting, and limited pixel area at `imgsz=416` are
  reasonable test hypotheses; the repository contains no evidence that these
  conditions were or were not represented in training.

## Recommended next steps (not implemented)

### Model and data improvements

1. Recover the exact training data, `data_finetune.yaml`, source checkpoint
   provenance, training command/log, and class-ID mapping. Record dataset
   version/license and hashes before using it.
2. Audit and correct boxes and class IDs, then publish image and instance
   counts per class, split membership, and representative examples. Check for
   near-duplicate frames/images across train, validation, and test.
3. Create an independent driver/vehicle/recording-level evaluation split.
   Include verified empty frames, hard negatives, confusing cabin objects,
   and representative small/occluded objects; review lighting, viewpoint,
   distance, scale, and occlusion coverage.
4. Report per-class precision/recall/AP and false positives by scenario and
   confidence range. Use validation data for any later model/threshold
   selection and reserve an untouched test set for final reporting.
5. If measured failures warrant it, expand/rebalance the training data and
   evaluate a new model candidate separately. Do not replace the current
   model or claim improvement without the independent test comparison.

### Software logic improvements

1. In a separately approved follow-up, require temporal confirmation for
   Cigarette, Cup, and Bottle detections before alert/audio/dashboard safety
   status; associate boxes to the same track rather than counting only class
   IDs.
2. Keep raw model boxes visibly distinct from confirmed detections and alert
   states. Do not let `dashboard_alert_status()` turn an unconfirmed raw Phone
   box into a possible-driver-use alert label.
3. Make dashboard, audio, and Telegram consume the same explicit confirmed
   detection/alert state. Retain the phone-specific driver confirmation,
   spatial association, and persistence semantics.
4. Keep model confidence filtering separate from temporal confirmation and
   avoid changing production thresholds until an independent validation
   protocol has measured the tradeoff.

These are design recommendations only; no behavior in the live application
was changed for this investigation.

## Regression tests required before a future fix

- **Per-class transient case:** one frame of each class must not produce a
  confirmed object alert; repeated detections of the same tracked object
  should confirm only after the chosen temporal criterion. A different object
  of the same class must not inherit another object's history.
- **Alert/audio/Telegram agreement:** raw, unconfirmed class 1/2/3 detections
  must not be sent, spoken, or rendered as confirmed alerts; confirmed
  detections should trigger each intended channel once, respect cooldown and
  active/clear transitions, and preserve Bottle's explicitly selected
  notification policy.
- **Phone consistency:** a single raw Phone box may remain a candidate/box,
  but must not be shown as possible driver use or trigger audio/Telegram until
  track confirmation, confirmed-driver proximity, and use/alert persistence
  all pass. Include passenger-side, missing-driver, stale-track, and
  intermittent-detection cases.
- **History semantics:** test that confidence filtering and temporal
  confirmation are distinct; if history remains, assert it validates matched
  object tracks, not only repeated class IDs.
- **Mapping and confidence boundaries:** retain tests for model-name order,
  class IDs, each per-class confidence boundary, maximum box area, and unknown
  class handling.
- **Dashboard fallback:** explicitly test `dashboard_alert_status()` with
  `alert_status == "NONE"` and raw-but-unconfirmed phone/non-phone detections.
- **Offline model evaluation:** with complete, reviewed per-frame annotations
  including verified-empty frames, verify raw and live-filtered per-class
  metrics, false positives, and independent split metadata. Do not use the
  existing placeholder `boxes.csv` as a fixture.

Existing focused tests include model mapping and adapter confidence behavior
(`tests/integration/test_existing_detection_compatibility.py`), phone track
validation (`tests/unit/test_temporal_detection.py`), and object-audio
activation (`tests/unit/test_alert_panel.py`). They do not currently cover all
the raw-versus-confirmed alert/dashboard paths listed above.
