# Face Recognition System - Complete Guide

## Overview
A complete face recognition pipeline that extracts faces from videos, creates an encoding database, and performs real-time face recognition using a webcam.

## Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

The system requires:
- `opencv-python` - For video processing and display
- `face_recognition` - For face detection and encoding
- `numpy` - For numerical operations
- `mediapipe`, `scikit-learn`, `ultralytics` - Other project libraries

### 2. Prepare Your Video Database
Create a `database/` folder in your project directory and add your video files.

**Important:** Video filenames should match person names (without extension):
```
database/
├── ranju.mp4       → Creates dataset/ranju/ folder
├── arun.mp4        → Creates dataset/arun/ folder
├── vijay.mp4       → Creates dataset/vijay/ folder
└── (add more videos as needed)
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.flv`

## How to Run

### Step-by-Step Usage

#### Option 1: Run All Steps Automatically
```bash
python face_recognition_system.py
```
Select option `4` - This will:
1. Extract frames from all videos
2. Create face encoding database
3. Start real-time face recognition

#### Option 2: Run Individual Steps
```bash
python face_recognition_system.py
```

Then select:
- **Option 1**: Extract frames from videos only
- **Option 2**: Create face encoding database only
- **Option 3**: Start real-time face recognition only

### Step 1: Extract Frames from Videos
**What happens:**
- Reads all videos from `database/` folder
- Creates folders in `dataset/` with person names
- Extracts frames every 10 frames
- Only saves frames that contain detected faces

**Output:**
```
dataset/
├── ranju/
│   ├── ranju_0000.jpg
│   ├── ranju_0001.jpg
│   └── ... (only frames with faces)
├── arun/
│   └── ... (face frames)
└── vijay/
    └── ... (face frames)
```

**Time:** Depends on video length (typically 30 seconds to 5 minutes per video)

### Step 2: Create Face Encoding Database
**What happens:**
- Reads all images from `dataset/` folder
- Detects faces in each image
- Extracts 128-dimensional face encodings
- Saves encodings to `faces.pkl`

**Output:**
```
faces.pkl (binary file containing all face encodings and names)
```

**Time:** 1-5 minutes depending on number of frames

### Step 3: Real-Time Face Recognition
**What happens:**
- Opens your webcam
- Detects faces in real-time
- Compares with saved encodings
- Shows person's name if match found
- Shows "Unknown" if no match
- Displays FPS (frames per second)

**Features:**
- ✓ Green bounding box = Known person (with confidence score)
- ✓ Red bounding box = Unknown person
- ✓ FPS counter (top-left)
- ✓ Confidence percentage for matches
- ✓ Press 'q' to quit

## File Structure

```
your_project/
├── face_recognition_system.py     # Main script
├── database/                        # Input videos (your video files)
│   ├── ranju.mp4
│   ├── arun.mp4
│   └── ...
├── dataset/                         # Extracted frames (auto-created)
│   ├── ranju/
│   ├── arun/
│   └── ...
├── faces.pkl                        # Face database (auto-created)
├── requirements.txt
└── face_recognition_system.py
```

## Key Functions

### `extract_frames_from_videos()`
Extracts frames from videos with face detection
- Reads from: `database/` folder
- Writes to: `dataset/person_name/` folder
- Parameter: `FRAME_EXTRACTION_INTERVAL = 10` (every 10th frame)

### `create_face_database()`
Creates face encodings database
- Reads from: `dataset/` folder
- Writes to: `faces.pkl`
- Uses: Deep learning face detector and encoder

### `recognize_faces_live()`
Real-time face recognition using webcam
- Reads from: `faces.pkl`
- Input: Webcam feed
- Output: Display with bounding boxes and names

## Configuration

Edit these constants in `face_recognition_system.py`:

```python
DATABASE_FOLDER = "database"                # Video storage folder
DATASET_FOLDER = "dataset"                 # Frame extraction folder
FACES_DATABASE_FILE = "faces.pkl"          # Encoding database file
FRAME_EXTRACTION_INTERVAL = 10             # Extract every Nth frame
FACE_DETECTION_TOLERANCE = 0.6             # Lower = stricter matching
RESIZE_SCALE = 0.25                        # Resize for faster processing
```

## Troubleshooting

### "No video files found"
- ✓ Check `database/` folder exists
- ✓ Check video file extensions (.mp4, .avi, etc.)
- ✓ Check video files are readable

### "No face encodings created"
- ✓ Ensure frames were extracted properly (Step 1)
- ✓ Check `dataset/` folder has person subfolders
- ✓ Try re-running Step 1

### "Face database not found"
- ✓ Run Step 2 first to create `faces.pkl`
- ✓ Ensure Step 1 completed successfully

### "Cannot open webcam"
- ✓ Check camera is connected
- ✓ Check no other app is using the camera
- ✓ Try unplugging and replugging camera

### Low Recognition Accuracy
- ✓ Use more video samples (longer videos)
- ✓ Ensure videos have clear face visibility
- ✓ Adjust `FACE_DETECTION_TOLERANCE` (increase for less strict matching)

## Performance Tips

1. **Faster Processing:**
   - Reduce video quality/length
   - Decrease `FRAME_EXTRACTION_INTERVAL` (extract fewer frames)
   - Use GPU if available

2. **Better Accuracy:**
   - Use longer videos (3+ minutes)
   - Ensure good lighting in videos
   - Include different angles and expressions
   - Use more video samples

3. **Memory Usage:**
   - Process videos in batches
   - Delete old `dataset/` folder before re-processing
   - Clear `__pycache__/` folder

## Example Workflow

```bash
# 1. Create project folder
mkdir face_recognition_project
cd face_recognition_project

# 2. Create database folder and add videos
mkdir database/
# Add ranju.mp4, arun.mp4, vijay.mp4 to database/

# 3. Install dependencies
pip install face_recognition opencv-python numpy

# 4. Run the system
python face_recognition_system.py

# 5. Select option 4 (Run All Steps)
# ... wait for processing to complete ...

# 6. Face recognition will start automatically after database is created
# 7. Press 'q' to quit
```

## Advanced Usage

### Running Individual Steps Programmatically

```python
from face_recognition_system import (
    extract_frames_from_videos,
    create_face_database,
    recognize_faces_live
)

# Step 1: Extract frames
extract_frames_from_videos()

# Step 2: Create database
create_face_database()

# Step 3: Start recognition
recognize_faces_live()
```

### Customize Parameters

```python
# Modify constants before running
FRAME_EXTRACTION_INTERVAL = 15  # Extract every 15 frames (faster)
FACE_DETECTION_TOLERANCE = 0.5  # Stricter matching
RESIZE_SCALE = 0.5              # Larger frames (slower but more accurate)
```

## System Requirements

- **RAM:** 2GB minimum (4GB+ recommended)
- **GPU:** Optional (CPU will work, but slower)
- **Webcam:** Required for real-time recognition
- **Python:** 3.7+
- **OS:** Windows, macOS, Linux

## Notes

- First run will be slow (downloads face_recognition models)
- Face encodings are stored in binary pickle format
- Confidence score ranges from 0.0 to 1.0 (higher = more confident)
- System works best with clear facial images

## License

Free to use and modify for personal projects.

---

**Need Help?** Check the comments in `face_recognition_system.py` for detailed explanations of each function.
