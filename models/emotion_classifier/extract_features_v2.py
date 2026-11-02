
from pathlib import Path
import argparse
import csv

import cv2
import mediapipe as mp
import numpy as np


CLASSES = [
    "angry", "confused", "disgust", "fear", "happy",
    "neutral", "sad", "shy", "surprise"
]

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)

SELECTED_POINTS = [
    33, 133, 160, 158, 153, 144,
    362, 263, 385, 387, 373, 380,
    107, 105, 70, 336, 334, 300,
    61, 291, 13, 14,
    1, 4, 5, 6, 10, 152
]


def distance(a, b):
    return float(np.linalg.norm(
        np.asarray(a)[:2] - np.asarray(b)[:2]
    ))


def calculate_ear(points, indices):
    p1, p2, p3, p4, p5, p6 = [
        points[i][:2] for i in indices
    ]

    horizontal = distance(p1, p4)

    if horizontal < 1e-8:
        raise ValueError("Invalid eye width")

    return (
        distance(p2, p6) + distance(p3, p5)
    ) / (2.0 * horizontal)


def extract_features(landmarks, image_width, image_height):

    # Convert normalized MediaPipe coordinates to pixel coordinates.
    # This prevents image aspect ratio from distorting facial geometry.
    points = np.asarray(
        [
            [
                p.x * image_width,
                p.y * image_height
            ]
            for p in landmarks
        ],
        dtype=np.float32
    )

    if len(points) < 468:
        raise ValueError("Insufficient facial landmarks")

    # Interocular distance in pixels.
    interocular = distance(points[33], points[263])

    if interocular < 1e-6:
        raise ValueError("Invalid interocular distance")

    # Eye geometry.
    left_ear = calculate_ear(points, LEFT_EYE)
    right_ear = calculate_ear(points, RIGHT_EYE)

    # Mouth geometry.
    mouth_width = distance(points[61], points[291])
    mouth_opening = distance(points[13], points[14])

    # Eyebrow-to-eye geometry.
    left_brow_eye = distance(points[107], points[159])
    right_brow_eye = distance(points[336], points[386])

    # Normalized geometric features.
    features = {
        "left_ear": left_ear,
        "right_ear": right_ear,
        "ear_difference": left_ear - right_ear,

        "mouth_aspect_ratio": (
            mouth_opening / max(mouth_width, 1e-8)
        ),

        "mouth_width_norm": mouth_width / interocular,
        "mouth_opening_norm": mouth_opening / interocular,

        "mouth_corner_delta_y": (
            points[291][1] - points[61][1]
        ) / interocular,

        "left_brow_eye_norm": left_brow_eye / interocular,
        "right_brow_eye_norm": right_brow_eye / interocular,

        "brow_eye_difference": (
            left_brow_eye - right_brow_eye
        ) / interocular,

        "left_brow_slope": (
            points[107][1] - points[70][1]
        ) / interocular,

        "right_brow_slope": (
            points[336][1] - points[300][1]
        ) / interocular,
    }

    # Preserve the selected facial landmark geometry.
    selected = points[SELECTED_POINTS, :2].copy()

    selected -= selected.mean(axis=0, keepdims=True)
    selected /= interocular

    for i, (x, y) in enumerate(selected):
        features[f"landmark_{i}_x"] = float(x)
        features[f"landmark_{i}_y"] = float(y)

    values = np.asarray(list(features.values()), dtype=np.float64)

    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite feature values")

    return features


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failures", required=True)

    args = parser.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output)
    failures_output = Path(args.failures)

    if not dataset.is_dir():
        raise SystemExit(f"Dataset not found: {dataset}")

    mp_face_mesh = mp.solutions.face_mesh

    rows = []
    failures = []

    print("Starting FANE feature extraction...")

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        for emotion in CLASSES:

            folder = dataset / emotion

            if not folder.is_dir():
                print(f"[WARNING] Missing folder: {emotion}")
                continue

            images = sorted([
                p for p in folder.iterdir()
                if p.suffix.lower() in (
                    ".jpg", ".jpeg", ".png",
                    ".bmp", ".webp"
                )
            ])

            success = 0
            failed = 0

            print(f"\nProcessing {emotion}: {len(images)} images")

            for image_path in images:

                image = cv2.imread(str(image_path))

                if image is None:
                    failures.append({
                        "image_path": str(image_path),
                        "label": emotion,
                        "reason": "image_read_failed"
                    })
                    failed += 1
                    continue

                height, width = image.shape[:2]

                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                )

                blur_score = float(
                    cv2.Laplacian(
                        gray,
                        cv2.CV_64F
                    ).var()
                )

                rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                result = face_mesh.process(rgb)

                if not result.multi_face_landmarks:
                    failures.append({
                        "image_path": str(image_path),
                        "label": emotion,
                        "reason": "face_not_detected"
                    })
                    failed += 1
                    continue

                try:

                    landmarks = (
                        result.multi_face_landmarks[0].landmark
                    )

                    # Pass actual image dimensions to the extractor.
                    features = extract_features(
                        landmarks,
                        width,
                        height
                    )

                    rows.append({
                        "image_path": str(image_path),
                        "label": emotion,
                        "image_width": width,
                        "image_height": height,
                        "blur_score": blur_score,
                        **features
                    })

                    success += 1

                except Exception as error:

                    failures.append({
                        "image_path": str(image_path),
                        "label": emotion,
                        "reason": str(error)
                    })

                    failed += 1

            print(
                f"{emotion}: extracted={success}, failed={failed}"
            )

    # Create output directories.
    output.parent.mkdir(parents=True, exist_ok=True)
    failures_output.parent.mkdir(parents=True, exist_ok=True)

    # Save extracted features.
    if rows:

        with output.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0].keys())
            )

            writer.writeheader()
            writer.writerows(rows)

    # Save extraction failures.
    with failures_output.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "label",
                "reason"
            ]
        )

        writer.writeheader()
        writer.writerows(failures)

    print("\n========== FINAL SUMMARY ==========")
    print(f"Successfully extracted: {len(rows)}")
    print(f"Failed images: {len(failures)}")
    print(f"Feature file: {output}")
    print(f"Failure file: {failures_output}")


if __name__ == "__main__":
    main()
