# Driver-Anomly-detection
Driver Anomaly Detection using Deep Learning for Accident Prevention

## Current application

The repository contains a Python desktop driver-monitoring prototype. The new
`src/driver_monitor/` package provides the supported entry points, profile data
validation/persistence, isolated emotion inference, and shared runtime helpers. The
packaged buffered runtime is used by both the new entry point and the Phase 3 compatibility
launcher. The monitor continues to use one camera feed and the existing recognition,
behavior/gaze, YOLO, and alert behavior.

Emotion is informational only. DeepFace's `happy` class is shown as
`smiling (estimated from happy)`, not as a direct smile measurement or a claim about internal
emotional state. Labels are disabled and reported as `Uncertain` until a versioned,
identity-separated evaluation produces a policy that passes the 0.75 precision gate on both
validation and held-out data. Emotion does not affect safety alerts, behavior-model inputs,
profiles, notifications, or logs, and is cleared when the live session stops.

## Setup

Use Python 3.11. From the repository root, create and activate a virtual environment, then
install the project extras:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[monitor,recognition,emotion]"
```

On Windows, the `recognition` extra may need Visual Studio 2022 Build Tools with the
**Desktop development with C++** workload: `face-recognition` depends on `dlib`, which pip may
need to build from source. This requires the MSVC v143 x64/x86 tools and a Windows 10/11 SDK.
Without a working compiler, the full install can fail. See
[docs/driver-monitoring-setup.md](docs/driver-monitoring-setup.md) for the compiler setup and
troubleshooting details.

The emotion extra installs TensorFlow and DeepFace and can require substantial download and disk
space. The combined extras are not yet validated as a clean, cross-platform install: DeepFace and
its `retina-face` dependency request OpenCV distributions that overlap the monitor's contrib build.
After installing, verify `cv2` and MediaPipe imports and test on the target computer before using a
webcam. The install command above describes the requested dependencies; it is not a guarantee that
the complete environment has been validated on every operating system.

Start live monitoring, including the optional aggregate timing summary:

```powershell
python -m driver_monitor --benchmark --width 960 --height 540
```

Open the driver profile manager:

```powershell
python -m driver_monitor --profiles
```

Run deterministic checks without a camera or real driver records:

```powershell
python -m unittest discover -s tests -v
```

See [docs/driver-monitoring-setup.md](docs/driver-monitoring-setup.md) for configuration,
evaluation, privacy boundaries, and migration notes.

## Existing models and code

Canonical behavior, object, and face-landmarker artifacts are organized under `models/` and
resolved there by the package and buffered monitor. Original phase-folder copies remain in place
for legacy scripts and reproducibility; their contents were hash-checked against the canonical
copies. Legacy runtime scripts are still being migrated incrementally, and the profile UI continues
to use the legacy profile database when present. No driver database or photo is included as a test
fixture. Model and dataset provenance still needs to be verified by the project owner before
public redistribution or release.
