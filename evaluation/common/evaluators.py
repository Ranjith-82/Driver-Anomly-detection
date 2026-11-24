from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from evaluation.common.inputs import EvaluationInputError, verify_independent_split


def _event_matches(
    truths: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[int, int]:
    candidates: dict[int, list[tuple[float, int]]] = {}
    for truth_index, truth in enumerate(truths):
        overlaps = []
        for prediction_index, prediction in enumerate(predictions):
            overlap = min(truth["end_seconds"], prediction["end_seconds"]) - max(
                truth["start_seconds"], prediction["start_seconds"]
            )
            if overlap > 0:
                overlaps.append((overlap, prediction_index))
        candidates[truth_index] = sorted(
            overlaps,
            key=lambda item: (
                -item[0],
                abs(predictions[item[1]]["start_seconds"] - truths[truth_index]["start_seconds"]),
                item[1],
            ),
        )

    assigned: dict[int, int] = {}

    def assign(truth_index: int, seen: set[int]) -> bool:
        for _, prediction_index in candidates[truth_index]:
            if prediction_index in seen:
                continue
            seen.add(prediction_index)
            previous_truth = assigned.get(prediction_index)
            if previous_truth is None or assign(previous_truth, seen):
                assigned[prediction_index] = truth_index
                return True
        return False

    for truth_index in sorted(candidates, key=lambda index: len(candidates[index])):
        assign(truth_index, set())
    return {truth_index: prediction_index for prediction_index, truth_index in assigned.items()}


def evaluate_fatigue_events(
    dataset_root: Path,
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score temporal fatigue events with one-to-one overlap matching."""
    events_path = Path(settings["events_csv"])
    model_path = Path(settings["model_path"])
    split_path = Path(settings["split_metadata"])
    for path, description in (
        (events_path, "Event manifest"),
        (model_path, "Fatigue model"),
        (split_path, "Independent split metadata"),
    ):
        if not path.is_file():
            raise EvaluationInputError(f"{description} does not exist: {path}")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    with events_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "clip_id", "identity_id", "sha256", "model_sha256", "source",
            "start_seconds", "end_seconds",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise EvaluationInputError(f"Event CSV is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise EvaluationInputError("Fatigue event manifest contains no rows.")
    verify_independent_split(split_path, rows)

    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    clip_identities: dict[str, str] = {}
    clip_hashes: dict[str, str] = {}
    clip_durations: dict[str, float] = {}
    for line_number, row in enumerate(rows, start=2):
        clip = (row.get("clip_id") or "").strip()
        identity = (row.get("identity_id") or "").strip()
        sample_hash = (row.get("sha256") or "").strip().lower()
        source = (row.get("source") or "").strip()
        if not clip or not identity or source not in {
            "ground_truth", "prediction", "negative_clip"
        }:
            raise EvaluationInputError(
                f"Invalid clip_id, identity_id, or source on event row {line_number}"
            )
        if len(sample_hash) != 64 or any(
            character not in "0123456789abcdef" for character in sample_hash
        ):
            raise EvaluationInputError(
                f"Invalid sha256 on event row {line_number}"
            )
        if source == "prediction":
            given_hash = (row.get("model_sha256") or "").strip().lower()
            if given_hash != model_hash:
                raise EvaluationInputError(
                    f"Prediction model hash does not match model_path on row {line_number}"
                )
        elif (row.get("model_sha256") or "").strip():
            raise EvaluationInputError(
                f"Only prediction rows may declare model_sha256 (row {line_number})"
            )

        if clip in clip_identities and clip_identities[clip] != identity:
            raise EvaluationInputError(
                f"Clip {clip!r} has inconsistent identity_id values."
            )
        if clip in clip_hashes and clip_hashes[clip] != sample_hash:
            raise EvaluationInputError(f"Clip {clip!r} has inconsistent sha256 values.")
        clip_identities[clip] = identity
        clip_hashes[clip] = sample_hash

        duration_value = (row.get("clip_duration_seconds") or "").strip()
        if duration_value:
            try:
                duration = float(duration_value)
            except ValueError as error:
                raise EvaluationInputError(
                    f"Invalid clip_duration_seconds on event row {line_number}"
                ) from error
            if not math.isfinite(duration) or duration <= 0:
                raise EvaluationInputError(
                    f"clip_duration_seconds must be finite and positive on row {line_number}"
                )
            if clip in clip_durations and clip_durations[clip] != duration:
                raise EvaluationInputError(
                    f"Clip {clip!r} has inconsistent duration values."
                )
            clip_durations[clip] = duration

        if source in {"ground_truth", "prediction"}:
            try:
                start = float(row["start_seconds"])
                end = float(row["end_seconds"])
            except (TypeError, ValueError) as error:
                raise EvaluationInputError(
                    f"Invalid event timestamps on row {line_number}"
                ) from error
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end <= start
            ):
                raise EvaluationInputError(
                    f"Event row {line_number} requires finite start_seconds < end_seconds"
                )
            groups[clip][source].append({
                **row,
                "start_seconds": start,
                "end_seconds": end,
                "line_number": line_number,
            })
        else:
            groups[clip][source].append({**row, "line_number": line_number})

    for clip, sources in groups.items():
        duration = clip_durations.get(clip)
        if duration is None:
            continue
        for source in ("ground_truth", "prediction"):
            for event in sources[source]:
                if event["end_seconds"] > duration:
                    raise EvaluationInputError(
                        f"Event on row {event['line_number']} ends after clip "
                        f"{clip!r} duration ({duration:.6f}s)."
                    )

    event_tp = event_fp = event_fn = 0
    clip_tp = clip_fp = clip_tn = clip_fn = 0
    delays: list[float] = []
    matched_rows: list[dict[str, Any]] = []
    missed_events: list[dict[str, Any]] = []
    for clip, sources in groups.items():
        truths = sources["ground_truth"]
        predictions = sources["prediction"]
        negative_markers = sources["negative_clip"]
        if len(negative_markers) > 1:
            raise EvaluationInputError(
                f"Clip {clip!r} has more than one negative_clip marker."
            )
        if negative_markers and truths:
            raise EvaluationInputError(
                f"Clip {clip!r} is both negative and contains ground-truth events."
            )
        if not truths and not negative_markers:
            raise EvaluationInputError(
                f"Clip {clip!r} needs ground_truth events or a negative_clip marker."
            )

        matches = _event_matches(truths, predictions)
        prediction_to_truth = {prediction: truth for truth, prediction in matches.items()}
        event_tp += len(matches)
        event_fn += len(truths) - len(matches)
        event_fp += len(predictions) - len(matches)
        if truths:
            if matches:
                clip_tp += 1
            else:
                clip_fn += 1
        elif predictions:
            clip_fp += 1
        else:
            clip_tn += 1

        for truth_index, truth in enumerate(truths):
            if truth_index not in matches:
                missed_events.append(truth)
                matched_rows.append({
                    "clip_id": clip,
                    "match": "false_negative",
                    "ground_truth": truth,
                })
                continue
            prediction = predictions[matches[truth_index]]
            delays.append(max(
                0.0, prediction["start_seconds"] - truth["start_seconds"]
            ))
            matched_rows.append({
                "clip_id": clip,
                "match": "true_positive",
                "ground_truth": truth,
                "prediction": prediction,
            })
        for prediction_index, prediction in enumerate(predictions):
            if prediction_index not in prediction_to_truth:
                matched_rows.append({
                    "clip_id": clip,
                    "match": "false_positive",
                    "prediction": prediction,
                })

    precision = event_tp / (event_tp + event_fp) if event_tp + event_fp else 0.0
    recall = event_tp / (event_tp + event_fn) if event_tp + event_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    clip_count = clip_tp + clip_fp + clip_tn + clip_fn
    duration_total = sum(clip_durations.values())
    has_complete_clip_durations = len(clip_durations) == len(groups)
    report = {
        "status": "evaluated",
        "dataset": settings.get("dataset", str(dataset_root)),
        "dataset_version": settings.get("dataset_version", ""),
        "model_sha256": model_hash,
        "sample_counts": {
            "clips": len(groups),
            "ground_truth_events": event_tp + event_fn,
            "predicted_events": event_tp + event_fp,
            "matched_events": event_tp,
            "missed_events": len(missed_events),
        },
        "metrics": {
            "true_positive": event_tp,
            "false_positive": event_fp,
            "false_negative": event_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_detection_delay_seconds": (
                sum(delays) / len(delays) if delays else 0.0
            ),
            "missed_event_count": len(missed_events),
            "duplicate_or_unmatched_prediction_count": event_fp,
            "event_false_positive_rate": None,
            "event_false_negative_rate": (
                event_fn / (event_fn + event_tp) if event_fn + event_tp else 0.0
            ),
            "false_alerts_per_minute": (
                event_fp / (duration_total / 60.0)
                if has_complete_clip_durations and duration_total else None
            ),
            "clip_level": {
                "true_positive": clip_tp,
                "true_negative": clip_tn,
                "false_positive": clip_fp,
                "false_negative": clip_fn,
                "accuracy": (
                    (clip_tp + clip_tn) / clip_count if clip_count else 0.0
                ),
                "false_positive_rate": (
                    clip_fp / (clip_fp + clip_tn) if clip_fp + clip_tn else 0.0
                ),
                "false_negative_rate": (
                    clip_fn / (clip_fn + clip_tp) if clip_fn + clip_tp else 0.0
                ),
            },
            "duration_coverage_seconds": duration_total if duration_total else None,
            "false_alerts_per_minute_status": (
                "computed" if has_complete_clip_durations and duration_total
                else "unavailable: clip_duration_seconds must be present for every clip"
            ),
        },
        "missed_events": missed_events,
        "metric_units": (
            "Event precision/recall/F1 count event matches and unmatched events; "
            "accuracy and false-positive/negative rates are clip-level."
        ),
    }
    return report, matched_rows
