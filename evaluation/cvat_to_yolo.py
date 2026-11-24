from __future__ import annotations

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.common.inputs import EvaluationInputError


BOX_FIELDS = (
    "video_name",
    "frame_index",
    "class_name",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
)


def _supported_labels() -> dict[str, str]:
    from driver_monitor.detection.objects import OBJECT_CLASSES

    return {name.casefold(): name for name in OBJECT_CLASSES.values()}


def _load_coverage(path: Path) -> dict[tuple[str, int], str]:
    coverage: dict[tuple[str, int], str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"video_name", "frame_index", "status"}
        if not required.issubset(reader.fieldnames or ()):
            raise EvaluationInputError(
                "Coverage CSV requires video_name,frame_index,status."
            )
        for line_number, row in enumerate(reader, start=2):
            video_name = Path((row["video_name"] or "").strip()).name
            try:
                frame_index = int(row["frame_index"])
            except (TypeError, ValueError) as error:
                raise EvaluationInputError(
                    f"Invalid frame_index on coverage row {line_number}."
                ) from error
            status = (row["status"] or "").strip().casefold()
            key = video_name, frame_index
            if not video_name or frame_index < 0:
                raise EvaluationInputError(
                    f"Invalid video_name or frame_index on coverage row {line_number}."
                )
            if status not in {"annotated", "verified_empty"}:
                raise EvaluationInputError(
                    f"Invalid coverage status on row {line_number}: "
                    "use annotated or verified_empty."
                )
            if key in coverage:
                raise EvaluationInputError(
                    f"Duplicate coverage entry for {video_name} frame {frame_index}."
                )
            coverage[key] = status
    return coverage


def _load_cvat_boxes(
    path: Path,
    video_name: str,
) -> dict[tuple[str, int], list[list[Any]]]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise EvaluationInputError(f"Invalid CVAT XML: {error}") from error
    if root.tag != "annotations" or any(root.iter("image")):
        raise EvaluationInputError(
            "Expected CVAT video-track XML with <annotations>/<track> entries."
        )

    valid_labels = _supported_labels()
    boxes: dict[tuple[str, int], list[list[Any]]] = defaultdict(list)
    for track in root.iter("track"):
        label = (track.get("label") or "").strip()
        class_name = valid_labels.get(label.casefold())
        if class_name is None:
            raise EvaluationInputError(
                f"Unsupported CVAT track label {label!r}; use the configured "
                "Phone, Cigarette, Cup, or Bottle class name."
            )
        seen_frames: set[int] = set()
        for box in track.findall("box"):
            try:
                frame_index = int(box.attrib["frame"])
                outside = box.attrib.get("outside", "0")
                coordinates = [
                    float(box.attrib[name])
                    for name in ("xtl", "ytl", "xbr", "ybr")
                ]
            except (KeyError, ValueError) as error:
                raise EvaluationInputError(
                    f"Invalid CVAT box in track {track.get('id', '<unknown>')}."
                ) from error
            if frame_index < 0 or outside not in {"0", "1"}:
                raise EvaluationInputError("CVAT frame and outside values are invalid.")
            if frame_index in seen_frames:
                raise EvaluationInputError(
                    f"Duplicate box for track {track.get('id', '<unknown>')} "
                    f"at frame {frame_index}."
                )
            seen_frames.add(frame_index)
            if outside == "1":
                continue
            x_min, y_min, x_max, y_max = coordinates
            if (
                not all(math.isfinite(value) and value >= 0 for value in coordinates)
                or x_max <= x_min
                or y_max <= y_min
            ):
                raise EvaluationInputError(
                    f"Invalid xyxy coordinates in CVAT track "
                    f"{track.get('id', '<unknown>')} at frame {frame_index}."
                )
            boxes[(video_name, frame_index)].append([
                class_name,
                x_min,
                y_min,
                x_max,
                y_max,
            ])
    return dict(boxes)


def convert_cvat_video_xml(
    xml_path: Path,
    coverage_path: Path,
    output_path: Path,
) -> dict[str, int]:
    """Convert CVAT track boxes using explicit per-frame annotation coverage."""
    coverage = _load_coverage(coverage_path)
    video_names = {video_name for video_name, _ in coverage}
    if len(video_names) != 1:
        raise EvaluationInputError(
            "Coverage CSV must describe frames from exactly one video."
        )
    video_name = next(iter(video_names))
    boxes = _load_cvat_boxes(xml_path, video_name)
    uncovered_boxes = sorted(set(boxes) - set(coverage))
    if uncovered_boxes:
        _, frame_index = uncovered_boxes[0]
        raise EvaluationInputError(
            f"CVAT contains a box for uncovered {video_name} frame {frame_index}."
        )

    rows: list[list[Any]] = []
    empty_frames = 0
    for (covered_video, frame_index), status in sorted(coverage.items()):
        frame_boxes = boxes.get((covered_video, frame_index), [])
        if status == "annotated" and not frame_boxes:
            raise EvaluationInputError(
                f"{covered_video} frame {frame_index} is marked annotated but "
                "has no visible CVAT boxes; mark it verified_empty only if reviewed."
            )
        if status == "verified_empty":
            if frame_boxes:
                raise EvaluationInputError(
                    f"{covered_video} frame {frame_index} is verified_empty but "
                    "has CVAT boxes."
                )
            rows.append([covered_video, frame_index, "", "", "", "", ""])
            empty_frames += 1
            continue
        rows.extend(
            [covered_video, frame_index, *box]
            for box in frame_boxes
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(BOX_FIELDS)
        writer.writerows(rows)
    return {
        "covered_frames": len(coverage),
        "object_boxes": len(rows) - empty_frames,
        "verified_empty_frames": empty_frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert CVAT video-track boxes with explicit frame coverage."
    )
    parser.add_argument("--xml", required=True, type=Path)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = convert_cvat_video_xml(args.xml, args.coverage, args.output)
    print(
        f"Wrote {result['object_boxes']} object boxes across "
        f"{result['covered_frames']} covered frames, including "
        f"{result['verified_empty_frames']} verified-empty frames."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
