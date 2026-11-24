from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def classification_metrics(
    actual: Sequence[str],
    predicted: Sequence[str],
    classes: Sequence[str],
) -> dict[str, Any]:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted counts differ")
    declared = list(dict.fromkeys(classes))
    if not declared:
        raise ValueError("at least one class is required")
    if any(label not in declared for label in (*actual, *predicted)):
        raise ValueError("actual or predicted labels contain an undeclared class")
    confusion = {label: {guess: 0 for guess in declared} for label in declared}
    for truth, guess in zip(actual, predicted):
        confusion[truth][guess] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    supports = {label: sum(confusion[label].values()) for label in declared}
    for label in declared:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in declared if other != label)
        fn = sum(confusion[label][other] for other in declared if other != label)
        total = len(actual)
        tn = total - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
            "actual_count": supports[label],
            "predicted_count": sum(confusion[row][label] for row in declared),
            "support": supports[label],
        }

    count = len(actual)
    accuracy = sum(
        confusion[label][label] for label in declared
    ) / count if count else 0.0
    macro_precision = sum(item["precision"] for item in per_class.values()) / len(declared)
    macro_recall = sum(item["recall"] for item in per_class.values()) / len(declared)
    macro_f1 = sum(item["f1"] for item in per_class.values()) / len(declared)
    weighted_precision = (
        sum(float(per_class[label]["precision"]) * supports[label] for label in declared)
        / count
        if count else 0.0
    )
    weighted_recall = (
        sum(float(per_class[label]["recall"]) * supports[label] for label in declared)
        / count
        if count else 0.0
    )
    weighted_f1 = (
        sum(float(per_class[label]["f1"]) * supports[label] for label in declared)
        / count
        if count else 0.0
    )
    return {
        "samples": count,
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "actual_class_counts": supports,
        "predicted_class_counts": {
            label: sum(confusion[row][label] for row in declared)
            for label in declared
        },
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def regression_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> dict[str, float | int]:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted counts differ")
    errors = [float(guess) - float(truth) for truth, guess in zip(actual, predicted)]
    if not all(math.isfinite(value) for value in errors):
        raise ValueError("regression values must be finite")
    count = len(errors)
    mean_error = sum(errors) / count if count else 0.0
    mae = sum(abs(error) for error in errors) / count if count else 0.0
    rmse = math.sqrt(sum(error * error for error in errors) / count) if count else 0.0
    return {"samples": count, "mean_error": mean_error, "mae": mae, "rmse": rmse}


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _average_precision(tp: Sequence[int], fp: Sequence[int], positives: int) -> float:
    if positives == 0:
        return 0.0
    cumulative_tp: list[int] = []
    cumulative_fp: list[int] = []
    tp_count = fp_count = 0
    for true_positive, false_positive in zip(tp, fp):
        tp_count += true_positive
        fp_count += false_positive
        cumulative_tp.append(tp_count)
        cumulative_fp.append(fp_count)
    recalls = [value / positives for value in cumulative_tp]
    precisions = [
        true / (true + false) if true + false else 0.0
        for true, false in zip(cumulative_tp, cumulative_fp)
    ]
    return sum(
        max(
            (precision for recall, precision in zip(recalls, precisions) if recall >= point),
            default=0.0,
        )
        for point in (index / 100 for index in range(101))
    ) / 101


def detection_metrics(
    samples: Sequence[Mapping[str, Any]],
    classes: Mapping[int, str],
) -> dict[str, Any]:
    """Compute box precision/recall at IoU .50 and COCO-style AP thresholds."""
    thresholds = [0.50 + step * 0.05 for step in range(10)]
    per_class: dict[str, Any] = {}
    ap_by_threshold: dict[float, list[float]] = defaultdict(list)
    total_tp = total_fp = total_fn = 0
    matched_ious: list[float] = []
    confusion_labels = list(classes.values()) + ["background"]
    confusion = {
        actual: {predicted: 0 for predicted in confusion_labels}
        for actual in confusion_labels
    }

    for class_id, class_name in classes.items():
        positives = sum(
            1
            for sample in samples
            for truth in sample["ground_truth"]
            if int(truth["class_id"]) == class_id
        )
        class_predictions = [
            (float(prediction["confidence"]), sample_index, prediction)
            for sample_index, sample in enumerate(samples)
            for prediction in sample["predictions"]
            if int(prediction["class_id"]) == class_id
        ]
        class_predictions.sort(key=lambda entry: entry[0], reverse=True)
        aps: list[float] = []
        for threshold in thresholds:
            matched: dict[int, set[int]] = defaultdict(set)
            true_positives: list[int] = []
            false_positives: list[int] = []
            for _, sample_index, prediction in class_predictions:
                candidates = [
                    (ground_index, _iou(prediction["box"], truth["box"]))
                    for ground_index, truth in enumerate(
                        samples[sample_index]["ground_truth"]
                    )
                    if int(truth["class_id"]) == class_id
                    and ground_index not in matched[sample_index]
                ]
                best_index, best_iou = max(candidates, key=lambda item: item[1], default=(-1, 0.0))
                is_match = best_index >= 0 and best_iou >= threshold
                true_positives.append(int(is_match))
                false_positives.append(int(not is_match))
                if is_match:
                    matched[sample_index].add(best_index)
            ap = _average_precision(true_positives, false_positives, positives)
            aps.append(ap)
            if positives:
                ap_by_threshold[threshold].append(ap)
        precision_matches: dict[int, set[int]] = defaultdict(set)
        class_tp = class_fp = 0
        class_matched_ious: list[float] = []
        for _, sample_index, prediction in class_predictions:
            candidates = [
                (ground_index, _iou(prediction["box"], truth["box"]))
                for ground_index, truth in enumerate(samples[sample_index]["ground_truth"])
                if int(truth["class_id"]) == class_id
                and ground_index not in precision_matches[sample_index]
            ]
            best_index, best_iou = max(candidates, key=lambda item: item[1], default=(-1, 0.0))
            if best_index >= 0 and best_iou >= 0.5:
                precision_matches[sample_index].add(best_index)
                class_tp += 1
                class_matched_ious.append(best_iou)
            else:
                class_fp += 1
        class_fn = positives - class_tp
        total_tp += class_tp
        total_fp += class_fp
        total_fn += class_fn
        per_class[class_name] = {
            "ground_truth": positives,
            "predictions": len(class_predictions),
            "true_positive": class_tp,
            "false_positive": class_fp,
            "false_negative": class_fn,
            "precision": class_tp / (class_tp + class_fp) if class_tp + class_fp else 0.0,
            "recall": class_tp / positives if positives else 0.0,
            "f1": (
                2 * class_tp / (2 * class_tp + class_fp + class_fn)
                if 2 * class_tp + class_fp + class_fn else 0.0
            ),
            "mean_iou_at_0_50": (
                sum(class_matched_ious) / len(class_matched_ious)
                if class_matched_ious else 0.0
            ),
            "ap50": aps[0],
            "ap50_95": sum(aps) / len(aps),
        }

    for sample in samples:
        remaining_ground = set(range(len(sample["ground_truth"])))
        remaining_prediction = set(range(len(sample["predictions"])))
        pairs = []
        for ground_index, truth in enumerate(sample["ground_truth"]):
            for prediction_index, prediction in enumerate(sample["predictions"]):
                if int(truth["class_id"]) != int(prediction["class_id"]):
                    continue
                pairs.append((
                    _iou(truth["box"], prediction["box"]),
                    ground_index,
                    prediction_index,
                ))
        pairs.sort(reverse=True)
        for overlap, ground_index, prediction_index in pairs:
            if (
                overlap >= 0.5
                and ground_index in remaining_ground
                and prediction_index in remaining_prediction
            ):
                truth_id = int(sample["ground_truth"][ground_index]["class_id"])
                prediction_id = int(sample["predictions"][prediction_index]["class_id"])
                confusion[classes[truth_id]][classes[prediction_id]] += 1
                matched_ious.append(overlap)
                remaining_ground.remove(ground_index)
                remaining_prediction.remove(prediction_index)
        for ground_index in remaining_ground:
            truth_id = int(sample["ground_truth"][ground_index]["class_id"])
            confusion[classes[truth_id]]["background"] += 1
        for prediction_index in remaining_prediction:
            prediction_id = int(sample["predictions"][prediction_index]["class_id"])
            confusion["background"][classes[prediction_id]] += 1

    mean_ap50 = (
        sum(ap_by_threshold[0.5]) / len(ap_by_threshold[0.5])
        if ap_by_threshold[0.5] else 0.0
    )
    mean_ap50_95 = (
        sum(sum(values) / len(values) for values in ap_by_threshold.values())
        / len(ap_by_threshold)
        if ap_by_threshold else 0.0
    )
    return {
        "images": len(samples),
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "precision": total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
        "recall": total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
        "f1": (
            2 * total_tp / (2 * total_tp + total_fp + total_fn)
            if 2 * total_tp + total_fp + total_fn else 0.0
        ),
        "mean_iou_at_0_50": (
            sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
        ),
        "map50": mean_ap50,
        "map50_95": mean_ap50_95,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "confusion_labels": confusion_labels,
    }
