
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd

from extract_features_v2 import extract_features


MODEL_DIR = Path("models/emotion_classifier")

MODEL_PATH = MODEL_DIR / "best_emotion_model.pkl"
FEATURE_PATH = MODEL_DIR / "feature_names.pkl"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Feature names not found: {FEATURE_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)

    if len(feature_names) != 68:
        raise ValueError(
            f"Expected 68 features, found {len(feature_names)}"
        )

    return model, feature_names


def predict_emotion(image_path):

    model, feature_names = load_model()

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Could not read image.")

    height, width = image.shape[:2]

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        result = face_mesh.process(rgb_image)

        if not result.multi_face_landmarks:
            raise ValueError("No face detected in the image.")

        landmarks = result.multi_face_landmarks[0].landmark

        # Use the exact training feature extraction function.
        features = extract_features(
            landmarks,
            width,
            height
        )

    # Preserve the exact training feature order.
    feature_vector = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names
    )

    if feature_vector.shape != (1, 68):
        raise ValueError(
            f"Unexpected feature shape: {feature_vector.shape}"
        )

    if not np.all(np.isfinite(feature_vector)):
        raise ValueError("Invalid feature values.")

    prediction = model.predict(feature_vector)[0]

    print("\n========== EMOTION PREDICTION ==========")
    print("Image:", image_path.name)
    print("Predicted emotion:", prediction)
    print("Features extracted:", feature_vector.shape[1])

    # This SVM was not trained with probability=True.
    # Its output must not be presented as a calibrated probability.
    print("Prediction confidence: Not calibrated")

    return prediction


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Standalone facial emotion inference"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input face image"
    )

    args = parser.parse_args()

    try:
        predict_emotion(args.image)

    except Exception as error:
        print("\nInference failed:", error)
