"""
Live Driver Face Recognition System
Recognizes drivers from the driver database and displays their details
"""

import cv2
import face_recognition
import numpy as np
import pickle
import os
from pathlib import Path
import time


class LiveDriverRecognition:
    def __init__(self, drivers_db='drivers_database.pkl'):
        """Initialize the live driver recognition system"""
        self.drivers_db = Path(drivers_db)
        self.base_dir = self.drivers_db.parent.resolve()
        self.photos_folder = self.base_dir / 'driver_photos'
        self.drivers = {}
        self.known_encodings = []
        self.known_driver_ids = []

        # Load data
        self.load_drivers()
        self.create_face_encodings()

    def load_drivers(self):
        """Load driver database"""
        if not os.path.exists(self.drivers_db):
            print("❌ Driver database not found!")
            print(f"   Please run driver_app.py first to create the database")
            return False

        try:
            with open(self.drivers_db, 'rb') as f:
                self.drivers = pickle.load(f)
            print(f"✓ Loaded {len(self.drivers)} driver(s)")
            return True
        except Exception as e:
            print(f"❌ Failed to load driver database: {e}")
            return False

    def create_face_encodings(self):
        """Create face encodings from driver photos"""
        print("\nCreating face encodings from driver photos...")

        encoding_count = 0
        failed_count = 0

        for driver_id, driver_info in self.drivers.items():
            photo_path = driver_info.get("photo")

            if not photo_path:
                print(f"⚠ No photo for driver {driver_id}")
                continue
            photo_path = Path(photo_path)
            if not photo_path.is_absolute():
                photo_path = self.base_dir / photo_path
            if not photo_path.exists():
                print(
                    f"⚠ Photo not found for driver {driver_id}: {photo_path}")
                continue

            try:
                # Load image
                image = face_recognition.load_image_file(photo_path)

                # Get face encodings
                face_locations = face_recognition.face_locations(image)

                if not face_locations:
                    print(f"⚠ No face detected in photo for {driver_id}")
                    failed_count += 1
                    continue

                face_encodings = face_recognition.face_encodings(
                    image, face_locations)

                if face_encodings:
                    # Use the first (or most prominent) face
                    self.known_encodings.append(face_encodings[0])
                    self.known_driver_ids.append(driver_id)
                    encoding_count += 1
                    print(
                        f"✓ Encoded face for {driver_id} - {driver_info.get('full_name', 'N/A')}")

            except Exception as e:
                print(f"✗ Error processing photo for {driver_id}: {e}")
                failed_count += 1

        print(f"\n✓ Created {encoding_count} face encoding(s)")
        if failed_count > 0:
            print(f"⚠ Failed to process {failed_count} driver(s)")

        return encoding_count > 0

    def format_driver_details(self, driver_id):
        """Format driver details for display"""
        if driver_id not in self.drivers:
            return None

        driver = self.drivers[driver_id]
        details = []

        details.append(f"ID: {driver.get('user_id', 'N/A')}")
        details.append(f"Name: {driver.get('full_name', 'N/A')}")
        details.append(f"Blood: {driver.get('blood_group', 'N/A')}")
        details.append(f"Phone: {driver.get('phone', 'N/A')}")
        details.append(f"License: {driver.get('license_number', 'N/A')}")
        details.append(f"City: {driver.get('city', 'N/A')}")

        return details

    def draw_driver_details(self, frame, driver_id, confidence, x, y, w, h):
        """Draw driver details on frame"""
        if driver_id not in self.drivers:
            return

        driver = self.drivers[driver_id]

        # Draw bounding box (green for recognized driver)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw name label with background
        name = driver.get('full_name', 'Unknown')
        label = f"{name} ({confidence:.2f})"

        label_y = y - 10 if y > 30 else y + h + 25
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)[0]

        # Background for text
        cv2.rectangle(
            frame,
            (x, label_y - 25),
            (x + text_size[0] + 10, label_y + 5),
            (0, 255, 0),
            -1
        )

        # Text
        cv2.putText(
            frame,
            label,
            (x + 5, label_y - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (0, 0, 0),
            1
        )

        # Draw details panel on top-right corner
        details = self.format_driver_details(driver_id)
        if details:
            panel_height = len(details) * 28 + 30
            panel_width = min(340, frame.shape[1] - 20)
            panel_x1 = frame.shape[1] - panel_width - 10
            panel_y1 = 10
            panel_x2 = frame.shape[1] - 10
            panel_y2 = 10 + panel_height

            # Panel background (semi-transparent)
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (panel_x1, panel_y1),
                (panel_x2, panel_y2),
                (10, 10, 10),
                -1
            )
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Panel border
            cv2.rectangle(
                frame,
                (panel_x1, panel_y1),
                (panel_x2, panel_y2),
                (0, 255, 0),
                2
            )

            # Panel title
            title = "Driver Details"
            title_size = cv2.getTextSize(
                title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            title_x = panel_x1 + 10
            title_y = panel_y1 + 25
            cv2.putText(frame, title, (title_x, title_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Details text
            text_x = panel_x1 + 10
            text_y = title_y + 30

            for detail in details:
                cv2.putText(
                    frame,
                    detail,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )
                text_y += 26

    def run_live_recognition(self, tolerance=0.5, resize_scale=0.25):
        """
        Run live face recognition with driver database

        Args:
            tolerance: Face matching tolerance (0.0-1.0, lower = stricter)
            resize_scale: Frame resize scale for faster processing
        """
        if not self.known_encodings:
            print("❌ No face encodings available!")
            print("   Please upload driver photos first")
            return

        print("\n" + "="*60)
        print("LIVE DRIVER FACE RECOGNITION")
        print("="*60)
        print(f"Loaded {len(self.known_encodings)} driver face encoding(s)")
        print("Press 'q' to quit")
        print("-" * 60 + "\n")

        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access webcam!")
            return

        # Optional: try to request a larger capture size for better display
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Create a resizable display window
        window_name = 'Live Driver Recognition - Press q to Quit'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 768)

        # FPS calculation
        fps_start = time.time()
        frame_count = 0
        process_this_frame = True

        # Recognition history (for smoothing)
        face_history = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            frame_count += 1
            h, w = frame.shape[:2]

            # Resize for faster processing
            small_frame = cv2.resize(
                frame, (0, 0), fx=resize_scale, fy=resize_scale)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Process every other frame
            if process_this_frame:
                # Detect faces
                face_locations = face_recognition.face_locations(
                    rgb_small_frame,
                    # Use HOG for speed (use 'cnn' for accuracy on GPU)
                    model='hog'
                )
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations)

                face_results = []

                for face_encoding, face_location in zip(face_encodings, face_locations):
                    # Compare with known faces
                    distances = face_recognition.face_distance(
                        self.known_encodings,
                        face_encoding
                    )

                    if len(distances) > 0:
                        best_match_index = np.argmin(distances)
                        best_distance = distances[best_match_index]

                        if best_distance < tolerance:
                            driver_id = self.known_driver_ids[best_match_index]
                            confidence = 1 - best_distance
                            face_results.append({
                                'driver_id': driver_id,
                                'confidence': confidence,
                                'location': face_location
                            })
                        else:
                            face_results.append({
                                'driver_id': None,
                                'confidence': 0,
                                'location': face_location
                            })
                    else:
                        face_results.append({
                            'driver_id': None,
                            'confidence': 0,
                            'location': face_location
                        })

                # Store results
                face_history.clear()
                for result in face_results:
                    face_history[len(face_history)] = result

            process_this_frame = not process_this_frame

            # Draw faces on frame
            for face_id, result in face_history.items():
                top, right, bottom, left = result['location']

                # Scale back to original size
                top = int(top / resize_scale)
                right = int(right / resize_scale)
                bottom = int(bottom / resize_scale)
                left = int(left / resize_scale)

                if result['driver_id']:
                    # Known driver - draw details
                    self.draw_driver_details(
                        frame,
                        result['driver_id'],
                        result['confidence'],
                        left, top,
                        right - left,
                        bottom - top
                    )
                else:
                    # Unknown face - red box
                    cv2.rectangle(frame, (left, top),
                                  (right, bottom), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        "Unknown",
                        (left, top - 10),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6,
                        (0, 0, 255),
                        1
                    )

            # FPS counter
            elapsed = time.time() - fps_start
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f} | Tolerance: {tolerance}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # Display frame
            cv2.imshow(window_name, frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Live recognition ended")


def main():
    """Main function"""
    print("\n" + "="*60)
    print("DRIVER LIVE RECOGNITION SYSTEM")
    print("="*60)

    # Initialize recognition system
    recognizer = LiveDriverRecognition()

    if not recognizer.drivers:
        print("\n❌ No drivers found in database!")
        print("   Please use driver_app.py to add drivers first")
        return

    if not recognizer.known_encodings:
        print("\n❌ No face encodings created!")
        print("   Please upload driver photos first")
        return

    print("\nConfiguration Options:")
    print("-" * 60)

    # Tolerance input
    print("\nFace matching tolerance (0.0-1.0):")
    print("  Lower = Stricter matching (more accurate)")
    print("  Higher = Looser matching (more false positives)")
    print("  Default = 0.5 (recommended)")

    try:
        tolerance_input = input(
            "Enter tolerance (press Enter for 0.5): ").strip()
        tolerance = float(tolerance_input) if tolerance_input else 0.5
        tolerance = max(0.0, min(1.0, tolerance))
    except ValueError:
        tolerance = 0.5

    print(f"\n✓ Using tolerance: {tolerance}")

    # Start live recognition
    recognizer.run_live_recognition(tolerance=tolerance)


if __name__ == "__main__":
    main()
