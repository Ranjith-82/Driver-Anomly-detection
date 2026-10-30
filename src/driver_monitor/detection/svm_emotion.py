from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SUPPORTED_EMOTIONS = (
    "angry",
    "confused",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "shy",
    "surprise",
)


class SVMEmotionAnalyzer:
    """Loads the trained SVM and its original feature extractor once."""

    def __init__(self, model_dir: Path) -> None:
        import cv2
        import joblib
        import mediapipe as mp
        import numpy as np
        import pandas as pd

        model_path = model_dir / "best_emotion_model.pkl"
        features_path = model_dir / "feature_names.pkl"
        metadata_path = model_dir / "model_metadata.pkl"
        for path in (model_path, features_path, metadata_path):
            if not path.is_file():
                raise FileNotFoundError(f"Emotion model file not found: {path}")

        self._model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        metadata = joblib.load(metadata_path)
        if (
            not isinstance(feature_names, (list, tuple))
            or len(feature_names) != 68
            or any(not isinstance(name, str) for name in feature_names)
            or len(set(feature_names)) != 68
        ):
            raise ValueError("Emotion feature names must be 68 unique strings")
        if not isinstance(metadata, dict) or metadata.get("feature_count") != 68:
            raise ValueError("Emotion model metadata has an invalid feature count")
        if tuple(metadata.get("classes", ())) != SUPPORTED_EMOTIONS:
            raise ValueError("Emotion model metadata has unsupported classes")
        if tuple(map(str, getattr(self._model, "classes_", ()))) != SUPPORTED_EMOTIONS:
            raise ValueError("Emotion model classes do not match model metadata")

        extractor_path = model_dir / "extract_features_v2.py"
        spec = importlib.util.spec_from_file_location(
            "_driver_monitor_extract_features_v2", extractor_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load emotion feature extractor: {extractor_path}")
        extractor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extractor_module)

        self._extract_features = extractor_module.extract_features
        self._feature_names = tuple(feature_names)
        self._np = np
        self._pd = pd
        self._cv2 = cv2
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def _feature_frame(self, features: Mapping[str, object]) -> Any:
        if len(features) != 68 or set(features) != set(self._feature_names):
            raise ValueError("Extracted emotion features do not match the model")
        ordered = [features[name] for name in self._feature_names]
        values = self._np.asarray(ordered, dtype=self._np.float64)
        if values.shape != (68,) or not self._np.all(self._np.isfinite(values)):
            raise ValueError("Emotion features contain invalid values")
        return self._pd.DataFrame([ordered], columns=self._feature_names)

    def predict(self, face_crop: Any) -> str | None:
        height, width = face_crop.shape[:2]
        rgb_image = self._cv2.cvtColor(face_crop, self._cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb_image)
        if not result.multi_face_landmarks:
            return None
        features = self._extract_features(
            result.multi_face_landmarks[0].landmark,
            width,
            height,
        )
        feature_frame = self._feature_frame(features)
        prediction = str(self._model.predict(feature_frame)[0]).casefold()
        return prediction if prediction in SUPPORTED_EMOTIONS else None

    def close(self) -> None:
        self._face_mesh.close()
