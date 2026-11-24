from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import torch
import yaml
import ultralytics
from ultralytics import YOLO


EXPECTED_PYTHON = Path(
    r"C:\Users\HP\miniconda3\envs\driver_friend_env\python.exe"
)
SOURCE_MODEL = Path(
    r"C:\Users\HP\Downloads\yolov8\runs\driver_objects"
    r"\four_class_final_v1\weights\best.pt"
)
DATASET = Path(
    r"C:\Users\HP\Downloads\yolov8\merged_driver_objects_hardneg_balanced_v1"
    r"\data.yaml"
)
RUNS_DIRECTORY = Path(
    r"C:\Users\HP\Downloads\yolov8\runs\driver_objects"
)
RUN_NAME = "balanced_hardneg_gpu_v1"
EXPECTED_NAMES = ("phone", "cigarette", "cup", "bottle")
IMAGE_SIZE = 416
VALIDATION_CONFIDENCE = 0.001
VALIDATION_IOU = 0.45
MAX_DETECTIONS = 20


def _verify_environment() -> None:
    actual_python = Path(sys.executable).resolve()
    if str(actual_python).casefold() != str(EXPECTED_PYTHON).casefold():
        raise RuntimeError(
            "Run this script with driver_friend_env: "
            f"expected {EXPECTED_PYTHON}, got {actual_python}"
        )
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"Python 3.11 is required, got {sys.version.split()[0]}"
        )
    if torch.__version__ != "2.14.1+cu130":
        raise RuntimeError(
            f"Expected PyTorch 2.14.1+cu130, got {torch.__version__}"
        )
    if ultralytics.__version__ != "8.4.33":
        raise RuntimeError(
            f"Expected Ultralytics 8.4.33, got {ultralytics.__version__}"
        )


def _normalized_names(names: object) -> tuple[str, ...]:
    if isinstance(names, dict):
        indexed = {int(key): str(value) for key, value in names.items()}
        return tuple(indexed[index].strip().casefold() for index in range(4))
    if isinstance(names, (list, tuple)):
        return tuple(str(value).strip().casefold() for value in names)
    raise TypeError(f"Unsupported YOLO class names: {type(names).__name__}")


def _verify_inputs() -> None:
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {SOURCE_MODEL}")
    if not DATASET.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATASET}")

    with DATASET.open(encoding="utf-8") as stream:
        dataset_config = yaml.safe_load(stream)
    actual_names = _normalized_names(dataset_config.get("names"))
    if actual_names != EXPECTED_NAMES or dataset_config.get("nc") != 4:
        raise ValueError(
            "Dataset class mapping must remain "
            f"{EXPECTED_NAMES}, got nc={dataset_config.get('nc')}, "
            f"names={actual_names}"
        )

    run_directory = RUNS_DIRECTORY / RUN_NAME
    if run_directory.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing run: {run_directory}"
        )


def _verify_cuda_model(model: YOLO) -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in this Python environment; "
            "refusing to start training."
        )

    torch.cuda.set_device(0)
    print(f"Training GPU: {torch.cuda.get_device_name(0)}")
    model.model.to("cuda:0")
    torch.cuda.synchronize(0)
    parameter = next(model.model.parameters())
    print(f"YOLO model parameter device: {parameter.device}")
    if parameter.device.type != "cuda" or parameter.device.index != 0:
        raise RuntimeError(
            f"YOLO model placement failed: expected cuda:0, got {parameter.device}"
        )
    print("Training device: 0 (CUDA GPU)")


def _class_metrics(model: YOLO, split_name: str) -> list[dict[str, object]]:
    metrics = model.val(
        data=str(DATASET),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=2,
        device=0,
        workers=2,
        cache=False,
        plots=False,
        verbose=True,
        conf=VALIDATION_CONFIDENCE,
        iou=VALIDATION_IOU,
        max_det=MAX_DETECTIONS,
        amp=True,
    )
    box_metrics = metrics.box
    class_ids = [int(value) for value in box_metrics.ap_class_index]
    rows = []
    for position, class_id in enumerate(class_ids):
        precision = float(box_metrics.p[position])
        recall = float(box_metrics.r[position])
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append(
            {
                "model": split_name,
                "class_id": class_id,
                "class_name": EXPECTED_NAMES[class_id].title(),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "map50": float(box_metrics.ap50[position]),
                "map50_95": float(box_metrics.ap[position]),
            }
        )
    if {int(row["class_id"]) for row in rows} != set(range(4)):
        raise RuntimeError(
            f"Validation did not produce metrics for all four classes: {rows}"
        )
    return rows


def _write_comparison(rows: list[dict[str, object]], output: Path) -> None:
    csv_path = output / "class_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = {
        model_name: {
            str(row["class_name"]): {
                key: row[key]
                for key in ("precision", "recall", "f1", "map50", "map50_95")
            }
            for row in rows
            if row["model"] == model_name
        }
        for model_name in ("Model B baseline", "Balanced hard-negative candidate")
    }
    comparison_path = output / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "dataset": str(DATASET),
                "validation": {
                    "image_size": IMAGE_SIZE,
                    "confidence": VALIDATION_CONFIDENCE,
                    "iou": VALIDATION_IOU,
                    "max_detections": MAX_DETECTIONS,
                    "device": "cuda:0",
                },
                "models": grouped,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Per-class metrics: {csv_path}")
    print(f"Model comparison: {comparison_path}")


def main() -> None:
    _verify_environment()
    _verify_inputs()
    model = YOLO(str(SOURCE_MODEL))
    if _normalized_names(model.names) != EXPECTED_NAMES:
        raise ValueError(
            f"Source checkpoint class mapping mismatch: {model.names}"
        )
    _verify_cuda_model(model)

    model.train(
        data=str(DATASET),
        epochs=6,
        patience=6,
        batch=2,
        imgsz=IMAGE_SIZE,
        device=0,
        workers=2,
        cache=False,
        optimizer="AdamW",
        lr0=0.0002,
        lrf=0.1,
        warmup_epochs=1,
        weight_decay=0.0005,
        seed=42,
        deterministic=True,
        pretrained=True,
        val=True,
        plots=False,
        verbose=True,
        conf=VALIDATION_CONFIDENCE,
        iou=VALIDATION_IOU,
        max_det=MAX_DETECTIONS,
        amp=True,
        close_mosaic=0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.005,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=1.0,
        translate=0.05,
        scale=0.2,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.0,
        auto_augment=None,
        project=str(RUNS_DIRECTORY),
        name=RUN_NAME,
        exist_ok=False,
    )

    output = RUNS_DIRECTORY / RUN_NAME
    best_checkpoint = output / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(
            f"Training completed without saving best checkpoint: {best_checkpoint}"
        )

    trained_model = YOLO(str(best_checkpoint))
    if _normalized_names(trained_model.names) != EXPECTED_NAMES:
        raise ValueError(
            f"Candidate checkpoint class mapping mismatch: {trained_model.names}"
        )
    rows = _class_metrics(YOLO(str(SOURCE_MODEL)), "Model B baseline")
    rows.extend(
        _class_metrics(trained_model, "Balanced hard-negative candidate")
    )
    _write_comparison(rows, output)
    print(f"Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
