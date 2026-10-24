
import cv2
import time
import torch
import numpy as np
from pathlib import Path
from collections import deque, Counter
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_phone_finetuned_v1.pt"

CAMERA_INDEX = 0
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
IMAGE_SIZE = 416

BASE_CONFIDENCE = 0.10
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 20
FRAME_SKIP = 2

# Model class IDs
CLASS_NAMES = {
    0: "Phone",
    1: "Cigarette",
    2: "Cup",
    3: "Bottle",
}

# Keep the existing thresholds
CLASS_CONFIDENCE = {
    0: 0.20,
    1: 0.50,
    2: 0.35,
    3: 0.40,
}

# OpenCV uses BGR colors
COLORS = {
    0: (30, 195, 255),
    1: (65, 75, 255),
    2: (65, 220, 125),
    3: (255, 175, 55),
}

BG = (20, 24, 32)
PANEL_BG = (32, 38, 48)
CARD_BG = (42, 49, 61)
WHITE = (250, 252, 255)
MUTED = (190, 200, 214)
ACCENT = (65, 200, 255)
GREEN = (80, 225, 140)

HISTORY_LENGTH = 3
MIN_HITS = 1

PANEL_WIDTH = 310
HEADER_HEIGHT = 70
FOOTER_HEIGHT = 42

WINDOW_NAME = "Driver Behaviour | Object Detection"

# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_text(
    image,
    text,
    x,
    y,
    scale=0.58,
    color=WHITE,
    thickness=1,
):
    cv2.putText(
        image,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_card(image, x, y, width, height):
    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        CARD_BG,
        -1,
    )


def draw_detection_box(frame, detection):
    x1, y1, x2, y2 = detection["box"]
    class_id = detection["class_id"]
    color = COLORS[class_id]

    label = (
        f'{detection["name"]} '
        f'{detection["confidence"]:.2f}'
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thickness = 2

    (text_width, text_height), baseline = (
        cv2.getTextSize(
            label,
            font,
            scale,
            thickness,
        )
    )

    label_top = max(0, y1 - text_height - 14)

    cv2.rectangle(
        frame,
        (x1, label_top),
        (
            x1 + text_width + 14,
            label_top + text_height + baseline + 10,
        ),
        color,
        -1,
    )

    draw_text(
        frame,
        label,
        x1 + 7,
        label_top + text_height + 4,
        scale,
        (15, 20, 27),
        thickness,
    )


# ============================================================
# DASHBOARD
# ============================================================

def build_dashboard(
    camera_frame,
    detections,
    loop_fps,
    inference_ms,
    inference_fps,
    device_name,
    frame_number,
):
    height, width = camera_frame.shape[:2]

    canvas_width = width + PANEL_WIDTH
    canvas_height = HEADER_HEIGHT + height + FOOTER_HEIGHT

    dashboard = np.zeros(
        (canvas_height, canvas_width, 3),
        dtype=np.uint8,
    )
    dashboard[:] = BG

    # ---------------- HEADER ----------------

    cv2.rectangle(
        dashboard,
        (0, 0),
        (canvas_width, HEADER_HEIGHT),
        PANEL_BG,
        -1,
    )

    cv2.rectangle(
        dashboard,
        (0, HEADER_HEIGHT - 3),
        (canvas_width, HEADER_HEIGHT),
        ACCENT,
        -1,
    )

    draw_text(
        dashboard,
        "DRIVER BEHAVIOUR",
        20, 30, 0.72, WHITE, 2,
    )

    draw_text(
        dashboard,
        "REAL-TIME OBJECT DETECTION",
        20, 53, 0.47, MUTED, 1,
    )

    cv2.circle(
        dashboard,
        (canvas_width - 75, 29),
        5,
        GREEN,
        -1,
    )

    draw_text(
        dashboard,
        "LIVE",
        canvas_width - 62, 35,
        0.52, GREEN, 2,
    )

    # ---------------- CAMERA VIEW ----------------

    camera_y = HEADER_HEIGHT

    dashboard[
        camera_y:camera_y + height,
        0:width,
    ] = camera_frame

    # ---------------- RIGHT PANEL ----------------

    panel_x = width

    cv2.rectangle(
        dashboard,
        (panel_x, HEADER_HEIGHT),
        (canvas_width, HEADER_HEIGHT + height),
        BG,
        -1,
    )

    x = panel_x + 14
    card_width = PANEL_WIDTH - 28
    y = camera_y + 25

    # Status heading
    draw_text(
        dashboard,
        "SYSTEM STATUS",
        x, y, 0.52, MUTED, 2,
    )

    detected_names = sorted({
        item["name"].upper()
        for item in detections
    })

    if detected_names:
        status = ", ".join(detected_names) + " DETECTED"
        status_color = ACCENT
    else:
        status = "SCANNING"
        status_color = GREEN

    # Status card
    status_y = y + 12

    draw_card(
        dashboard,
        x, status_y,
        card_width, 43,
    )

    cv2.circle(
        dashboard,
        (x + 12, status_y + 21),
        4,
        status_color,
        -1,
    )

    # Fit status text within the panel
    status_scale = 0.40
    if len(status) > 22:
        status_scale = 0.32

    draw_text(
        dashboard,
        status,
        x + 23,
        status_y + 27,
        status_scale,
        status_color,
        2,
    )

    # ---------------- PERFORMANCE ----------------

    perf_y = status_y + 60

    draw_text(
        dashboard,
        "PERFORMANCE",
        x, perf_y, 0.52, MUTED, 2,
    )

    metrics = [
        ("Loop FPS", f"{loop_fps:.1f}"),
        ("Inference", f"{inference_ms:.1f} ms"),
        ("YOLO rate", f"{inference_fps:.1f}/s"),
        ("Device", device_name),
    ]

    metric_top = perf_y + 12
    metric_height = 31
    metric_gap = 5

    for index, (label, value) in enumerate(metrics):
        cy = metric_top + index * (metric_height + metric_gap)

        draw_card(
            dashboard,
            x, cy,
            card_width, metric_height,
        )

        draw_text(
            dashboard,
            label,
            x + 8,
            cy + 21,
            0.43,
            MUTED,
            1,
        )

        draw_text(
            dashboard,
            value,
            x + 112,
            cy + 21,
            0.45,
            WHITE,
            2,
        )

    # ---------------- DETECTION SUMMARY ----------------
    # All four classes always remain visible.

    summary_y = metric_top + 4 * (metric_height + metric_gap) + 12

    draw_text(
        dashboard,
        "DETECTION SUMMARY",
        x, summary_y, 0.48, MUTED, 2,
    )

    counts = Counter(
        item["class_id"]
        for item in detections
    )

    row_top = summary_y + 10
    row_height = 35

    for index, class_id in enumerate(CLASS_NAMES):
        cy = row_top + index * row_height

        cv2.circle(
            dashboard,
            (x + 7, cy + 13),
            5,
            COLORS[class_id],
            -1,
        )

        draw_text(
            dashboard,
            CLASS_NAMES[class_id],
            x + 20,
            cy + 19,
            0.48,
            WHITE,
            1,
        )

        draw_text(
            dashboard,
            f'{counts.get(class_id, 0):02d}',
            x + card_width - 31,
            cy + 19,
            0.53,
            COLORS[class_id],
            2,
        )

    # ---------------- FOOTER ----------------

    footer_y = HEADER_HEIGHT + height

    cv2.rectangle(
        dashboard,
        (0, footer_y),
        (canvas_width, canvas_height),
        PANEL_BG,
        -1,
    )

    draw_text(
        dashboard,
        f"Frame: {frame_number}",
        12, footer_y + 27,
        0.43, MUTED, 1,
    )

    draw_text(
        dashboard,
        f"Active detections: {len(detections)}",
        width // 2,
        footer_y + 27,
        0.43, WHITE, 1,
    )

    draw_text(
        dashboard,
        "Q  EXIT",
        canvas_width - 76,
        footer_y + 27,
        0.43, WHITE, 2,
    )

    return dashboard


# ============================================================
# MODEL INITIALIZATION
# ============================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Keep best_phone_finetuned_v1.pt beside webcam.py."
    )

use_gpu = torch.cuda.is_available()
device = 0 if use_gpu else "cpu"
device_name = "CUDA GPU" if use_gpu else "CPU"

print("Loading model:", MODEL_PATH.name)
print("Inference device:", device_name)

model = YOLO(str(MODEL_PATH))
print("Model classes:", model.names)

if not all(
    class_id in model.names
    for class_id in CLASS_NAMES
):
    raise ValueError(
        f"Unexpected model classes: {model.names}"
    )


# ============================================================
# WEBCAM INITIALIZATION
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check CAMERA_INDEX "
        "and close other camera applications."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

history = deque(maxlen=HISTORY_LENGTH)
last_detections = []

frame_number = 0
loop_fps = 0.0
last_time = time.perf_counter()

inference_count = 0
inference_total_seconds = 0.0
inference_ms = 0.0
inference_fps = 0.0

# Fit the dashboard to a typical laptop display.
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1150, 700)

print("Dashboard started. Press Q to quit.")


# ============================================================
# MAIN LOOP
# ============================================================

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            print("Unable to read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_number += 1

        # Run inference every second frame
        if frame_number % FRAME_SKIP == 0:
            start = time.perf_counter()

            result = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=BASE_CONFIDENCE,
                iou=IOU_THRESHOLD,
                device=device,
                half=use_gpu,
                max_det=MAX_DETECTIONS,
                verbose=False,
            )[0]

            if use_gpu:
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            inference_count += 1
            inference_total_seconds += elapsed

            inference_ms = (
                inference_total_seconds
                / inference_count
                * 1000
            )

            inference_fps = (
                inference_count
                / inference_total_seconds
            )

            current = []

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())

                    if class_id not in CLASS_NAMES:
                        continue

                    if confidence < CLASS_CONFIDENCE[class_id]:
                        continue

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist(),
                    )

                    box_area = (
                        max(0, x2 - x1)
                        * max(0, y2 - y1)
                    )

                    frame_area = (
                        frame.shape[0]
                        * frame.shape[1]
                    )

                    if box_area > 0.85 * frame_area:
                        continue

                    current.append({
                        "class_id": class_id,
                        "name": CLASS_NAMES[class_id],
                        "confidence": confidence,
                        "box": (x1, y1, x2, y2),
                    })

            last_detections = current

            history.append({
                item["class_id"]
                for item in current
            })

        # Smooth display across skipped frames
        hit_counts = Counter(
            class_id
            for frame_classes in history
            for class_id in frame_classes
        )

        stable_classes = {
            class_id
            for class_id, hits in hit_counts.items()
            if hits >= MIN_HITS
        }

        visible = [
            item for item in last_detections
            if item["class_id"] in stable_classes
        ]

        for item in visible:
            draw_detection_box(frame, item)

        now = time.perf_counter()
        delta = now - last_time
        last_time = now

        if delta > 0:
            instantaneous_fps = 1.0 / delta

            loop_fps = (
                instantaneous_fps
                if loop_fps == 0
                else 0.85 * loop_fps
                + 0.15 * instantaneous_fps
            )

        dashboard = build_dashboard(
            camera_frame=frame,
            detections=visible,
            loop_fps=loop_fps,
            inference_ms=inference_ms,
            inference_fps=inference_fps,
            device_name=device_name,
            frame_number=frame_number,
        )

        cv2.imshow(WINDOW_NAME, dashboard)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    print("Webcam closed.")

    if inference_count > 0:
        print(
            "Average inference latency: "
            f"{inference_total_seconds / inference_count * 1000:.2f} ms"
        )

        print(
            "Average YOLO rate: "
            f"{inference_count / inference_total_seconds:.2f} calls/s"
        )
