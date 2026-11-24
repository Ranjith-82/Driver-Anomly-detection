from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def write_predictions_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["video_name", "frame_index", "timestamp_seconds"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_csv(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        def write_values(prefix: str, values: Mapping[str, Any]) -> None:
            for key, value in values.items():
                metric_name = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, Mapping):
                    write_values(metric_name, value)
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    writer.writerow([metric_name, value])

        write_values("", report)


def write_json(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_confusion_matrix_png(
    path: Path,
    matrix: Mapping[str, Mapping[str, int]],
    labels: Sequence[str] | None = None,
    *,
    column_labels: Sequence[str] | None = None,
) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "OpenCV and NumPy are required to render the confusion matrix."
        ) from error

    row_labels = list(labels or matrix)
    column_labels = list(column_labels or labels or matrix)
    cell_width = 150
    cell_height = 54
    left = 210
    top = 120
    width = left + cell_width * len(column_labels) + 20
    height = top + cell_height * (len(row_labels) + 1) + 20
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.putText(canvas, "Confusion Matrix", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 35, 45), 2)
    cv2.putText(canvas, "Predicted", (left + 10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 35, 45), 2)
    for column, label in enumerate(column_labels):
        x = left + column * cell_width
        cv2.putText(canvas, label[:18], (x + 7, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (25, 35, 45), 1)
    for row_index, actual_label in enumerate(row_labels):
        y = top + row_index * cell_height
        cv2.putText(canvas, actual_label[:22], (12, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 35, 45), 1)
        for column, predicted_label in enumerate(column_labels):
            count = int(matrix.get(actual_label, {}).get(predicted_label, 0))
            x = left + column * cell_width
            shade = max(0, 245 - min(count, 100) * 2)
            cv2.rectangle(canvas, (x, y), (x + cell_width - 2, y + cell_height - 2), (shade, 245, shade), -1)
            cv2.rectangle(canvas, (x, y), (x + cell_width - 2, y + cell_height - 2), (190, 195, 200), 1)
            cv2.putText(canvas, str(count), (x + 12, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 30, 40), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write confusion matrix image: {path}")
