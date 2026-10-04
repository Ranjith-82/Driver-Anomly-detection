import cv2
import time
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

COCO_MODEL = "yolov8n.pt"
CIGARETTE_MODEL = "cigarette_model.pt"

CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CIGARETTE_IMGSZ = 416
COCO_IMGSZ = 416

CIGARETTE_CONF = 0.55
COCO_CONF = 0.45

# Detection frequency
CIGARETTE_SKIP = 2
COCO_SKIP = 5


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("=" * 60)
print("Loading models...")
print("=" * 60)

coco_model = YOLO(COCO_MODEL)
cigarette_model = YOLO(CIGARETTE_MODEL)

print("COCO model loaded.")
print("Cigarette model loaded.")

print()
print("Models loaded successfully!")
print("=" * 60)


# ============================================================
# OPEN WEBCAM
# ============================================================

print()
print("Opening webcam...")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("ERROR: Cannot open webcam.")
    exit()


# ============================================================
# CAMERA SETTINGS
# ============================================================

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


# ============================================================
# VARIABLES
# ============================================================

frame_count = 0

cigarette_results = None
coco_results = None

previous_time = time.time()

fps = 0.0
fps_frame_count = 0


# ============================================================
# COCO DETECTION
#
# ONLY:
#   cell phone
#   cup
#   bottle
# ============================================================

def draw_coco_results(frame, results):

    phone_detected = False

    if results is None:
        return phone_detected


    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence < COCO_CONF:
                continue


            class_id = int(box.cls[0])

            class_name = coco_model.names[class_id]


            # ==================================================
            # ONLY 3 COCO OBJECTS
            # ==================================================

            allowed_classes = {
                "cell phone",
                "cup",
                "bottle"
            }


            if class_name not in allowed_classes:
                continue


            # ==================================================
            # COORDINATES
            # ==================================================

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ==================================================
            # PHONE = RED
            # CUP/BOTTLE = GREEN
            # ==================================================

            if class_name == "cell phone":

                box_color = (0, 0, 255)

                phone_detected = True

            else:

                box_color = (0, 255, 0)


            # ==================================================
            # DRAW BOX
            # ==================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )


            # ==================================================
            # LABEL
            # ==================================================

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )


            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                box_color,
                2
            )


    return phone_detected


# ============================================================
# CIGARETTE DETECTION
# ============================================================

def draw_cigarette_results(frame, results):

    cigarette_detected = False


    if results is None:
        return cigarette_detected


    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence < CIGARETTE_CONF:
                continue


            class_id = int(box.cls[0])

            class_name = cigarette_model.names[class_id]


            # ==================================================
            # COORDINATES
            # ==================================================

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ==================================================
            # CIGARETTE = RED
            # ==================================================

            box_color = (0, 0, 255)

            cigarette_detected = True


            # ==================================================
            # DRAW BOX
            # ==================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )


            # ==================================================
            # LABEL
            # ==================================================

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )


            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2
            )


    return cigarette_detected


# ============================================================
# WEBCAM LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print("Failed to read webcam frame.")

        break


    frame_count += 1
    fps_frame_count += 1


    # ========================================================
    # CIGARETTE MODEL
    # ========================================================

    if frame_count % CIGARETTE_SKIP == 0:

        cigarette_results = (
            cigarette_model.predict(
                source=frame,
                imgsz=CIGARETTE_IMGSZ,
                conf=CIGARETTE_CONF,
                device="cpu",
                verbose=False,
                max_det=5
            )
        )


    # ========================================================
    # COCO MODEL
    # ========================================================

    if frame_count % COCO_SKIP == 0:

        coco_results = (
            coco_model.predict(
                source=frame,
                imgsz=COCO_IMGSZ,
                conf=COCO_CONF,
                device="cpu",
                verbose=False,
                max_det=10
            )
        )


    # ========================================================
    # DRAW COCO
    # ========================================================

    phone_detected = draw_coco_results(
        frame,
        coco_results
    )


    # ========================================================
    # DRAW CIGARETTE
    # ========================================================

    cigarette_detected = draw_cigarette_results(
        frame,
        cigarette_results
    )


    # ========================================================
    # ALERT
    #
    # PHONE OR CIGARETTE
    # ========================================================

    if cigarette_detected:

        cv2.putText(
            frame,
            "ALERT: cigarette",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            3
        )

    elif phone_detected:

        cv2.putText(
            frame,
            "ALERT: cell phone",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            3
        )


    # ========================================================
    # FPS CALCULATION
    # ========================================================

    current_time = time.time()

    elapsed = (
        current_time -
        previous_time
    )


    if elapsed >= 1.0:

        fps = (
            fps_frame_count /
            elapsed
        )

        fps_frame_count = 0

        previous_time = current_time


    # ========================================================
    # FPS ONLY
    #
    # BOTTOM LEFT
    # ========================================================

    height, width = frame.shape[:2]

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (15, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "YOLOv8 Object Detection",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("=" * 60)
print("Webcam closed.")
print("=" * 60)