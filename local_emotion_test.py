
from deepface import DeepFace
import cv2

EMOTIONS = [
    "angry", "disgust", "fear", "happy",
    "sad", "surprise", "neutral"
]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit(1)

print("Webcam opened. Press Q to quit.")

frame_count = 0
emotion_scores = {}
dominant_emotion = "Starting..."
error_message = ""

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            print("ERROR: Could not read webcam frame.")
            break

        frame_count += 1

        # Run inference periodically to keep the preview responsive.
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

                emotion_scores = result.get("emotion", {})
                dominant_emotion = result.get(
                    "dominant_emotion", "No result"
                )
                error_message = ""

                print("\nDominant:", dominant_emotion)
                for name in EMOTIONS:
                    score = emotion_scores.get(name)
                    if score is not None:
                        print(f"{name:8s}: {score:7.2f}")

            except Exception as error:
                error_message = str(error)[:70]
                print("Emotion error:", error_message)

        # Display the top prediction.
        cv2.putText(
            frame,
            f"Top: {dominant_emotion}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # Display scores for all seven classes.
        for index, name in enumerate(EMOTIONS):
            score = emotion_scores.get(name)
            text = f"{name}: {score:.1f}" if score is not None else f"{name}: N/A"

            cv2.putText(
                frame,
                text,
                (20, 70 + index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
            )

        if error_message:
            cv2.putText(
                frame,
                "Inference error - check terminal",
                (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Emotion Diagnostic Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
