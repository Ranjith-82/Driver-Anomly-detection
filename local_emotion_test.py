from deepface import DeepFace
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit(1)

print("Webcam opened. Press Q to quit.")
frame_count = 0
emotion_label = "Starting..."

try:
    while True:
        if not ok:
            print("ERROR: Could not read webcam frame.")
            break

        frame_count += 1

        # Run inference periodically so the webcam preview stays responsive.
        if frame_count % 15 == 1:
            try:
                result = DeepFace.analyze(
                    img_path=frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                    silent=True,
                )
                if isinstance(result, list):
                    result = result[0] if result else {}
                emotion_label = result.get("dominant_emotion", "No result")
            except Exception as error:
                emotion_label = "Error: " + str(error)[:45]
                print(emotion_label)

        cv2.putText(
            frame,
            "Emotion: " + emotion_label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Local Emotion Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
