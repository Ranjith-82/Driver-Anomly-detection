from __future__ import annotations

from pathlib import Path
from typing import Any


EXPECTED_EMOTIONS = frozenset({
    "angry",
    "confused",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "shy",
    "surprise",
})


class CNNEmotionAnalyzer:
    """Loads the trained MobileNetV3-Small checkpoint once for inference."""

    def __init__(self, checkpoint_path: Path) -> None:
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import models, transforms

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Emotion checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("Emotion checkpoint must contain a dictionary")

        raw_classes = checkpoint.get("classes")
        if (
            not isinstance(raw_classes, (list, tuple))
            or any(not isinstance(label, str) for label in raw_classes)
            or len(raw_classes) != len(EXPECTED_EMOTIONS)
            or len(set(raw_classes)) != len(raw_classes)
            or set(raw_classes) != EXPECTED_EMOTIONS
        ):
            raise ValueError("Emotion checkpoint has an invalid class mapping")
        self.classes = tuple(raw_classes)

        image_size = checkpoint.get("image_size")
        if (
            isinstance(image_size, bool)
            or not isinstance(image_size, int)
            or image_size != 160
        ):
            raise ValueError("Emotion checkpoint must use 160x160 input images")

        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Emotion checkpoint is missing model_state_dict")

        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features,
            len(self.classes),
        )
        model.load_state_dict(state_dict, strict=True)

        self._torch = torch
        self._np = np
        self._image = Image
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = model.to(self.device)
        self._model.eval()
        self._transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, face_crop: Any) -> tuple[str, float]:
        if (
            not hasattr(face_crop, "shape")
            or len(face_crop.shape) != 3
            or face_crop.shape[2] != 3
            or face_crop.dtype != self._np.uint8
            or face_crop.size == 0
        ):
            raise ValueError("Emotion inference requires a non-empty BGR face crop")

        rgb_crop = self._np.ascontiguousarray(face_crop[:, :, ::-1])
        image = self._image.fromarray(rgb_crop)
        input_tensor = self._transform(image).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            logits = self._model(input_tensor)
            if tuple(logits.shape) != (1, len(self.classes)):
                raise ValueError("Emotion model returned an invalid output shape")
            probabilities = self._torch.softmax(logits, dim=1)
            confidence, class_index = probabilities[0].max(dim=0)

        score = float(confidence.item())
        if not 0.0 <= score <= 1.0:
            raise ValueError("Emotion model returned an invalid confidence score")
        return self.classes[int(class_index.item())], score
