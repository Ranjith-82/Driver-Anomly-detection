from __future__ import annotations

from pathlib import Path
from typing import Any


OBJECT_CLASSES = {
    0: "Phone",
    1: "Cigarette",
    2: "Cup",
    3: "Bottle",
}


class YoloObjectDetector:
    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                f"YOLO runtime dependency is unavailable: {error.name}"
            ) from error
        self._torch = torch
        self._model = YOLO(str(model_path))
        self._validate_class_names(self._model.names)

    @staticmethod
    def _validate_class_names(names: Any) -> None:
        actual = (
            {int(key): str(value) for key, value in names.items()}
            if isinstance(names, dict)
            else dict(enumerate(map(str, names)))
        )
        for class_id, expected in OBJECT_CLASSES.items():
            if actual.get(class_id, "").strip().casefold() != expected.casefold():
                raise ValueError(
                    f"Incompatible YOLO class mapping for {class_id}: "
                    f"expected {expected!r}, got {actual.get(class_id)!r}"
                )

    def detect(
        self,
        frame,
        *,
        image_size: int = 416,
        confidence: float = 0.1,
        iou: float = 0.45,
        max_detections: int = 20,
        class_confidence: dict[int, float] | None = None,
        max_box_fraction: float = 1.0,
    ):
        use_gpu = self._torch.cuda.is_available()
        result = self._model.predict(
            source=frame,
            imgsz=image_size,
            conf=confidence,
            iou=iou,
            device=0 if use_gpu else "cpu",
            half=use_gpu,
            max_det=max_detections,
            verbose=False,
        )[0]
        detections = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            if class_id not in OBJECT_CLASSES:
                continue
            score = float(box.conf[0].item())
            if (
                class_confidence is not None
                and score < class_confidence.get(class_id, 1.0)
            ):
                continue
            coordinates = tuple(int(value) for value in box.xyxy[0].tolist())
            x1, y1, x2, y2 = coordinates
            frame_area = frame.shape[0] * frame.shape[1]
            box_area = max(0, x2 - x1) * max(0, y2 - y1)
            if box_area > max_box_fraction * frame_area:
                continue
            detections.append({
                "class_id": class_id,
                "name": OBJECT_CLASSES[class_id],
                "confidence": score,
                "box": coordinates,
            })
        return detections
