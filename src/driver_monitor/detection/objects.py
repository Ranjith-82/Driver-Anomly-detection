from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from driver_monitor.monitoring.timing import log_timing


OBJECT_CLASSES = {
    0: "Phone",
    1: "Cigarette",
    2: "Cup",
    3: "Bottle",
}


class YoloObjectDetector:
    def __init__(self, model_path: Path) -> None:
        log_timing("YOLO_MODEL_LOADING_START", model=model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        import_started = time.perf_counter()
        try:
            import torch
            log_timing(
                "YOLO_TORCH_IMPORT_COMPLETE",
                duration=f"{time.perf_counter() - import_started:.3f}s",
            )
            import_started = time.perf_counter()
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                f"YOLO runtime dependency is unavailable: {error.name}"
            ) from error
        log_timing(
            "YOLO_ULTRALYTICS_IMPORT_COMPLETE",
            duration=f"{time.perf_counter() - import_started:.3f}s",
        )
        self._torch = torch
        model_started = time.perf_counter()
        self._model = YOLO(str(model_path))
        log_timing(
            "YOLO_MODEL_CONSTRUCTOR_COMPLETE",
            duration=f"{time.perf_counter() - model_started:.3f}s",
        )
        validation_started = time.perf_counter()
        self._validate_class_names(self._model.names)
        log_timing(
            "YOLO_CLASS_VALIDATION_COMPLETE",
            duration=f"{time.perf_counter() - validation_started:.3f}s",
        )
        log_timing("YOLO_DEVICE_INITIALIZATION_START")
        device_started = time.perf_counter()
        self._use_gpu = self._torch.cuda.is_available()
        if self._use_gpu:
            self._torch.cuda.init()
        log_timing(
            "YOLO_DEVICE_INITIALIZATION_COMPLETE",
            device="CUDA GPU" if self._use_gpu else "CPU",
            duration=f"{time.perf_counter() - device_started:.3f}s",
        )
        self._first_inference = True
        self._first_valid_detection = True
        self.last_inference_seconds = 0.0
        self.last_postprocess_seconds = 0.0
        log_timing(
            "YOLO_MODEL_READY",
            model=model_path.name,
            device="CUDA GPU" if self._use_gpu else "CPU",
        )

    @property
    def device_name(self) -> str:
        return "CUDA GPU" if self._use_gpu else "CPU"

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
        use_gpu = (
            self._use_gpu
            if hasattr(self, "_use_gpu")
            else self._torch.cuda.is_available()
        )
        first_inference = getattr(self, "_first_inference", False)
        inference_started = time.perf_counter()
        if first_inference:
            log_timing(
                "YOLO_FIRST_INFERENCE_START",
                image_size=image_size,
                device="CUDA GPU" if use_gpu else "CPU",
            )
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
        self.last_inference_seconds = time.perf_counter() - inference_started
        if first_inference:
            log_timing(
                "YOLO_FIRST_INFERENCE_COMPLETE",
                duration=f"{self.last_inference_seconds:.3f}s",
            )
            self._first_inference = False
        postprocess_started = time.perf_counter()
        detections = []
        if result.boxes is None:
            self.last_postprocess_seconds = (
                time.perf_counter() - postprocess_started)
            if first_inference:
                log_timing(
                    "YOLO_FIRST_POSTPROCESS_COMPLETE",
                    duration=f"{self.last_postprocess_seconds:.3f}s",
                    valid_detections=0,
                )
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
        if detections and getattr(self, "_first_valid_detection", False):
            log_timing(
                "YOLO_FIRST_VALID_DETECTION",
                frame_size=frame.shape[:2],
                class_id=detections[0]["class_id"],
                confidence=f'{detections[0]["confidence"]:.3f}',
            )
            self._first_valid_detection = False
        self.last_postprocess_seconds = time.perf_counter() - postprocess_started
        if first_inference:
            log_timing(
                "YOLO_FIRST_POSTPROCESS_COMPLETE",
                duration=f"{self.last_postprocess_seconds:.3f}s",
                valid_detections=len(detections),
            )
        return detections
