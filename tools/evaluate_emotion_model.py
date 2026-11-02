from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SOURCE_LABELS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)
SOURCE_TO_DISPLAY = {
    "angry": "anger",
    "happy": "smiling",
    "neutral": "neutral",
}
DISPLAY_LABELS = tuple(SOURCE_TO_DISPLAY.values())
DEFAULT_MINIMUM_PRECISION = 0.75


def normalize_ground_truth(label: str) -> str:
    normalized = label.strip().casefold()
    aliases = {"angry": "anger", "happy": "smiling"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in DISPLAY_LABELS:
        raise ValueError(f"Unsupported ground-truth emotion label: {label!r}")
    return normalized


def normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    identity_id = str(row.get("identity_id", "")).strip()
    partition = str(row.get("partition", "")).strip()
    if not identity_id:
        raise ValueError("Each evaluation row requires identity_id")
    if partition not in {"validation", "held-out"}:
        raise ValueError(f"Invalid evaluation partition: {partition!r}")
    scores = row.get("scores")
    if not isinstance(scores, Mapping):
        scores = {
            name: row.get(f"score_{name}")
            for name in SOURCE_LABELS
        }
    clean_scores: dict[str, float] = {}
    for source_label in SOURCE_LABELS:
        score = scores.get(source_label)
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError(
                f"Each row requires a finite score_{source_label} value"
            )
        clean_scores[source_label] = float(score)
    return {
        "identity_id": identity_id,
        "partition": partition,
        "label": normalize_ground_truth(str(row.get("label", ""))),
        "scores": clean_scores,
    }


def validate_partitions(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_validation_examples: int = 50,
    min_held_out_examples: int = 100,
    min_validation_identities: int = 5,
    min_held_out_identities: int = 10,
) -> None:
    normalized = [normalize_row(row) for row in rows]
    validation = [
        row for row in normalized if row["partition"] == "validation"]
    held_out = [row for row in normalized if row["partition"] == "held-out"]
    if not validation or not held_out:
        raise ValueError(
            "Both validation and held-out partitions are required")
    validation_ids = {row["identity_id"] for row in validation}
    held_out_ids = {row["identity_id"] for row in held_out}
    overlap = validation_ids & held_out_ids
    if overlap:
        raise ValueError(
            f"Identity leakage between partitions: {len(overlap)} identity/identities overlap"
        )
    for partition_name, partition_rows, minimum, min_identities in (
        ("validation", validation, min_validation_examples, min_validation_identities),
        ("held-out", held_out, min_held_out_examples, min_held_out_identities),
    ):
        identities = {row["identity_id"] for row in partition_rows}
        if len(identities) < min_identities:
            raise ValueError(
                f"{partition_name} partition needs at least {min_identities} identities"
            )
        counts = Counter(row["label"] for row in partition_rows)
        missing = {
            label: counts[label]
            for label in DISPLAY_LABELS
            if counts[label] < minimum
        }
        if missing:
            raise ValueError(
                f"{partition_name} partition needs at least {minimum} examples "
                f"per class; insufficient counts: {missing}"
            )


def _winning_source(scores: Mapping[str, float]) -> str:
    return max(SOURCE_LABELS, key=scores.__getitem__)


def select_class_thresholds(
    rows: Sequence[Mapping[str, Any]],
    minimum_precision: float = DEFAULT_MINIMUM_PRECISION,
) -> dict[str, float | None]:
    validation_rows = [
        normalize_row(row) for row in rows
        if row.get("partition") == "validation"
    ]
    result: dict[str, float | None] = {}
    for display_label in DISPLAY_LABELS:
        source_label = next(
            source for source, display in SOURCE_TO_DISPLAY.items()
            if display == display_label
        )
        candidates = sorted({
            row["scores"][source_label]
            for row in validation_rows
            if _winning_source(row["scores"]) == source_label
        })
        chosen = None
        for cutoff in candidates:
            predicted = [
                row for row in validation_rows
                if _winning_source(row["scores"]) == source_label
                and row["scores"][source_label] >= cutoff
            ]
            if not predicted:
                continue
            precision = sum(
                row["label"] == display_label for row in predicted) / len(predicted)
            if precision >= minimum_precision:
                chosen = cutoff
                break
        result[display_label] = chosen
    return result


def predict_display_label(
    scores: Mapping[str, float],
    thresholds: Mapping[str, float | None],
) -> str:
    source_label = _winning_source(scores)
    display_label = SOURCE_TO_DISPLAY.get(source_label)
    if display_label is None:
        return "uncertain"
    threshold = thresholds.get(display_label)
    if threshold is None or scores[source_label] < threshold:
        return "uncertain"
    return display_label


def evaluate_partition(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float | None],
) -> dict[str, Any]:
    normalized = [normalize_row(row) for row in rows]
    confusion = {
        actual: {predicted: 0 for predicted in (*DISPLAY_LABELS, "uncertain")}
        for actual in DISPLAY_LABELS
    }
    class_counts = {
        label: {"true_positive": 0, "predicted": 0, "actual": 0}
        for label in DISPLAY_LABELS
    }
    uncertain = 0
    for row in normalized:
        actual = row["label"]
        predicted = predict_display_label(row["scores"], thresholds)
        confusion[actual][predicted] += 1
        class_counts[actual]["actual"] += 1
        if predicted == "uncertain":
            uncertain += 1
        else:
            class_counts[predicted]["predicted"] += 1
            if actual == predicted:
                class_counts[predicted]["true_positive"] += 1
    class_metrics = {}
    for label, counts in class_counts.items():
        precision = (
            counts["true_positive"] / counts["predicted"]
            if counts["predicted"] else 0.0
        )
        recall = (
            counts["true_positive"] / counts["actual"]
            if counts["actual"] else 0.0
        )
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "actual_count": counts["actual"],
            "predicted_count": counts["predicted"],
        }
    return {
        "examples": len(normalized),
        "uncertain_rate": uncertain / len(normalized) if normalized else 0.0,
        "classes": class_metrics,
        "confusion_matrix": confusion,
    }


def load_and_score_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        from deepface import DeepFace
    except ImportError as error:
        raise RuntimeError(
            "DeepFace is required to evaluate image manifests; install the emotion extra."
        ) from error

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"image_path", "identity_id", "partition", "label"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Manifest image on line {line_number} does not exist: {image_path}"
                )
            result = DeepFace.analyze(
                img_path=str(image_path),
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                if len(result) != 1:
                    raise ValueError(
                        f"Expected one face for manifest line {line_number}, "
                        f"got {len(result)}"
                    )
                result = result[0]
            if not isinstance(result, dict) or not isinstance(result.get("emotion"), dict):
                raise ValueError(
                    f"DeepFace returned invalid emotion scores on line {line_number}"
                )
            rows.append(normalize_row({
                **row,
                "scores": result["emotion"],
            }))
    return rows, "DeepFace " + getattr(DeepFace, "__version__", "version not reported")


def build_report(
    rows: Sequence[Mapping[str, Any]],
    provenance: str,
    model_version: str,
) -> tuple[str, dict[str, Any]]:
    normalized = [normalize_row(row) for row in rows]
    validation = [
        row for row in normalized if row["partition"] == "validation"]
    held_out = [row for row in normalized if row["partition"] == "held-out"]
    thresholds = select_class_thresholds(validation)
    validation_metrics = evaluate_partition(validation, thresholds)
    held_out_metrics = evaluate_partition(held_out, thresholds)
    class_config: dict[str, dict[str, Any]] = {}
    for display_label, source_label in (
        ("anger", "angry"),
        ("smiling", "happy"),
        ("neutral", "neutral"),
    ):
        validation_precision = validation_metrics["classes"][display_label]["precision"]
        held_out_precision = held_out_metrics["classes"][display_label]["precision"]
        class_config[source_label] = {
            "threshold": thresholds[display_label],
            "validation_precision": validation_precision,
            "held_out_precision": held_out_precision,
            "enabled": (
                thresholds[display_label] is not None
                and validation_precision >= DEFAULT_MINIMUM_PRECISION
                and held_out_precision >= DEFAULT_MINIMUM_PRECISION
            ),
        }
    policy = {
        "evaluation_version": "driver-monitor-emotion-v1",
        "minimum_precision": DEFAULT_MINIMUM_PRECISION,
        "dataset_provenance": provenance,
        "model_version": model_version,
        "classes": class_config,
    }
    lines = [
        "# Emotion Model Evaluation",
        "",
        f"- Dataset provenance/license: {provenance}",
        f"- Model: {model_version}",
        f"- Validation examples: {validation_metrics['examples']}",
        f"- Held-out examples: {held_out_metrics['examples']}",
        f"- Validation identities: {len({row['identity_id'] for row in validation})}",
        f"- Held-out identities: {len({row['identity_id'] for row in held_out})}",
        "- Partitioning: identity-disjoint",
        "- Release gate: precision >= 0.75 on validation and held-out partitions",
        "",
        "## Per-class results",
        "",
        "| Display label | Source class | Cutoff | Validation precision | Held-out precision | Held-out recall | Enabled |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for display_label, source_label in (
        ("anger", "angry"),
        ("smiling (estimated from happy)", "happy"),
        ("neutral", "neutral"),
    ):
        item = class_config[source_label]
        recall = held_out_metrics["classes"][
            "smiling" if source_label == "happy" else display_label
        ]["recall"]
        lines.append(
            f"| {display_label} | {source_label} | "
            f"{item['threshold'] if item['threshold'] is not None else 'disabled'} | "
            f"{item['validation_precision']:.3f} | {item['held_out_precision']:.3f} | "
            f"{recall:.3f} | {'yes' if item['enabled'] else 'no'} |"
        )
    lines.extend([
        "",
        f"- Validation uncertain rate: {validation_metrics['uncertain_rate']:.3f}",
        f"- Held-out uncertain rate: {held_out_metrics['uncertain_rate']:.3f}",
        "",
        "## Held-out confusion matrix",
        "",
        "| Actual \\ Predicted | Anger | Smiling | Neutral | Uncertain |",
        "|---|---:|---:|---:|---:|",
    ])
    for actual in DISPLAY_LABELS:
        row = held_out_metrics["confusion_matrix"][actual]
        lines.append(
            f"| {actual} | {row['anger']} | {row['smiling']} | "
            f"{row['neutral']} | {row['uncertain']} |"
        )
    lines.extend([
        "",
        "## Limitations",
        "",
        "- Model scores are relative outputs, not calibrated probabilities or findings about internal emotional state.",
        "- The `happy` source class is shown only as the qualified `smiling (estimated from happy)` proxy.",
        "- Results apply only to the recorded dataset provenance and evaluation protocol.",
        "",
    ])
    return "\n".join(lines), policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate DeepFace emotion scores.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--thresholds-output", type=Path)
    parser.add_argument("--dataset-provenance", required=True)
    args = parser.parse_args(argv)

    rows, model_version = load_and_score_manifest(args.manifest)
    validate_partitions(rows)
    report, policy = build_report(rows, args.dataset_provenance, model_version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    thresholds_path = args.thresholds_output or args.output.with_suffix(
        ".thresholds.json")
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
