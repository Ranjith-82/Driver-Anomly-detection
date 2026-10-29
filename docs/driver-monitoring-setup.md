# Driver Monitoring Setup and Limitations

## Runtime

- Supported interpreter: Python 3.11 (`requires-python` is constrained to `<3.12`).
- Install from the repository root with
  `python -m pip install -e ".[monitor,recognition,emotion]"`.
- Launch the buffered monitor with `python -m driver_monitor`; `--benchmark` prints aggregate
  capture-to-render timings when the session exits.
- Telegram notifications are sent for supported driver-safety alerts and camera errors. In
  PowerShell, set the bot token and destination chat ID before launching the monitor:

  ```powershell
  $env:DRIVER_MONITOR_BOT_TOKEN = "your_bot_token"
  $env:DRIVER_MONITOR_CHAT_ID = "your_chat_id"
  python -m driver_monitor
  ```

  Keep the token private; do not commit it to the repository. Credentials are read from the
  current process environment and are not saved by the application.
- YOLO model loading begins on a separate worker while gaze calibration runs. Timestamped
  `[TIMING]` messages distinguish camera readiness/first frame, calibration, PyTorch and
  Ultralytics imports, model construction/class validation/device selection, first inference,
  and first detection. The worker reports initialization errors in the dashboard and console.
- Phone predictions remain visible as raw boxes, but phone presence is not treated as driver
  use. A phone must persist across frames before confirmation; a separate suspected-use state
  requires a confirmed driver identity and a persistent phone prediction near that driver's
  recognized face region. This is a conservative spatial heuristic, not proof of phone use.
  Configure these windows with `--phone-persistence-seconds`,
  `--driver-phone-suspicion-seconds`,
  `--driver-phone-alert-persistence-seconds`, and
  `--object-max-missed-frames`.
- Talking, yawning, and distraction behavior labels are debounced independently of the
  classifier. Fatigue remains immediate; the four behavior persistence options are
  `--talking-persistence-seconds`, `--yawning-persistence-seconds`,
  `--behavior-distraction-persistence-seconds`, and
  `--behavior-clear-persistence-seconds`.
- The live dashboard includes a driver-alert panel beside the camera feed. On Windows, new
  alerts use generated Windows waveform audio for beeps and one persistent Windows speech
  synthesizer running in the audio worker. Fatigue speech repeats no faster than every 45
  seconds; distraction starts after 1.5 continuous seconds and repeats a 900 Hz/180 ms beep
  and voice no faster than every 4 seconds; each yawn event receives two 650 Hz/120 ms beeps
  with a 120 ms gap, no faster than every 10 seconds. Talking is announced only after four
  separate detected episodes and no more than once every two minutes. Audio is disabled with
  `--mute`. Ensure Windows output-device and application volumes are enabled.
- Launch profile management with `python -m driver_monitor --profiles`.
- The packaged buffered runtime uses the Phase 3 script location for legacy algorithm modules
  and model/profile compatibility; the Phase 3 path is a thin wrapper and does not open a second
  camera loop.

### Windows: dlib compiler prerequisite

The `recognition` extra includes `face-recognition`, which depends on `dlib`. For this Python 3.11
Windows environment, pip builds dlib from source; it therefore needs a working Microsoft C++
compiler. If the build stops with `You must use Visual Studio to build a python extension on
windows` or `You have not installed Visual C++`, the C++ build tools are missing or not active in
the shell. Updating pip alone does not supply this compiler.

1. Install [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. In the Visual Studio Installer, select **Desktop development with C++** and include the
   **MSVC v143 x64/x86 build tools**, a **Windows 10 or 11 SDK**, and **C++ CMake tools for
   Windows**.
3. Close existing terminals. Open **Developer PowerShell for VS 2022** (or **x64 Native Tools
   Command Prompt for VS 2022**) so the compiler environment is initialized.
4. From the repository root, verify `cl` resolves, then build dlib on its own before retrying all
   extras:

```powershell
Set-Location "C:\path\to\Driver-Anomly-detection"
.\.venv\Scripts\Activate.ps1
Get-Command cl
python -m pip install --upgrade pip setuptools wheel
python -m pip install dlib
python -m pip install -e ".[monitor,recognition,emotion]"
```

If PowerShell activation is restricted, use the venv interpreter explicitly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install dlib
.\.venv\Scripts\python.exe -m pip install -e ".[monitor,recognition,emotion]"
```

The first `dlib` build is the focused check for the reported C++ toolchain error. If it still
fails, retain the complete CMake/compiler error and verify the VS C++ workload and that the shell
is the VS Developer PowerShell; do not install arbitrary third-party dlib wheels or silently
skip the dependency.

The emotion extra installs TensorFlow and DeepFace. A successful package install or mocked unit
test does not demonstrate camera compatibility, driver recognition, GPU support, or real-time
performance; verify these on the actual reference PC.
The combined extras are not yet release-validated: DeepFace and its `retina-face` dependency
request OpenCV distributions that overlap the monitor's contrib build. Resolve that packaging
overlap and verify `cv2`/MediaPipe imports in a clean Python 3.11 environment before treating the
combined installation as supported.

## Current component map

| Responsibility | Current location |
|---|---|
| Package entry point and configuration | `src/driver_monitor/` |
| Canonical behavior, object, and face-landmarker artifacts | `models/behavior/`, `models/objects/`, and `models/face_landmarker/` |
| Emotion mapping and DeepFace adapter | `src/driver_monitor/detection/emotion.py` |
| Buffered live camera, workers, and alerts | `src/driver_monitor/monitoring/legacy_runtime.py` |
| Existing facial behavior algorithm source | `Phase-2/Face_monitoring_system/`, invoked through `src/driver_monitor/detection/behavior.py` |
| Gaze algorithm source | `Phase-3/Buffered-system/buffered_system/Buffer/Buffered_system/`, invoked through `src/driver_monitor/detection/gaze.py` |
| YOLO inference adapter and class mapping | `src/driver_monitor/detection/objects.py` |
| Object and behavior temporal validation | `src/driver_monitor/monitoring/temporal.py` |
| Profile UI, validation, and repository | `src/driver_monitor/profiles/` and `src/driver_monitor/ui/driver_profiles.py` |
| Deterministic tests | `tests/unit/` and `tests/integration/` |

The package and buffered runtime resolve model artifacts from `models/`. The legacy phase folders
are retained with their original artifact copies for old scripts and reproducibility; their hashes
were checked against the canonical copies. The Phase 2 profile GUI class is retained in
`Phase-2/Face_recognition_system/driver_app_legacy.py`, while `driver_app.py` is its package UI
launcher.

## Model and output policy

The emotion adapter requests only DeepFace emotion analysis and crops the face region identified
by the existing face-landmark worker. It is an independent sampled worker; it cannot block the
camera or other detectors. The adapter does not save crops, scores, or histories. A result is
shown only when its class-specific threshold policy exists and both validation and identity-
disjoint held-out precision are at least 0.75. Missing policy, non-target labels, invalid results,
or below-threshold scores display `Uncertain`; missing faces and worker failures have separate
visible statuses.

The `happy` output is a proxy displayed as `smiling (estimated from happy)`. It is not a dedicated
smile detector, calibrated probability, or inference about a driver's internal mental state.
Emotion never enters the behavior-model feature vector or safety-alert rules.

The live application uses the existing driver database if it is present at
`Phase-2/Face_recognition_system/drivers_database.pkl`; otherwise it uses
`data/local/drivers_database.pkl`. The profile manager and face-recognition worker resolve the
same database path. Existing photo references are preserved. Locally captured/uploaded profile
photos are managed under the associated local photo directory. Delete actions remove a photo only
after confirmation and only when the path is under the managed photo directory.

The existing database uses Python pickle. Treat it as trusted local data only; do not open a
database from an unknown source. Never commit driver database files, photos, evaluation images,
or session output. The root `.gitignore` excludes local profile and evaluation paths.

## Emotion evaluation

Use only a versioned, appropriately licensed dataset with documented provenance and consent
status. Keep images and the identity-bearing manifest outside version control. The manifest uses
`image_path`, `identity_id`, `partition` (`validation` or `held-out`), and `label` columns. Labels
are `anger`, `happy`/`smiling`, and `neutral`. Use at least 50 validation examples per class
across 5 identities and at least 100 held-out examples per class across 10 different identities.
The tool rejects identity overlap and missing classes.

```powershell
python tools/evaluate_emotion_model.py `
  --manifest <licensed-dataset-manifest.csv> `
  --dataset-provenance "<dataset version and license>" `
  --output docs/validation/emotion-evaluation.md `
  --thresholds-output data/local/emotion-thresholds.json
```

The tool analyzes each image in memory, writes an aggregate Markdown report and a local threshold
policy, and does not write face crops or raw per-image scores. A class remains disabled unless
the lowest validation cutoff achieving 0.75 precision also gives at least 0.75 held-out
precision. Review dataset licensing and metrics before using the generated local policy.

## Performance and safety regression

For the SC-001 performance run, use 960x540 input, normal indoor lighting, a documented Windows
PC/camera, CPU-only baseline, 60 seconds of warm-up, and five minutes of steady state. The
benchmark is capture-to-dashboard-render using monotonic timestamps; at least 95% of changed
combined results must render within two seconds. The optional `--benchmark` output contains only
aggregate timing statistics and is not persisted. Record the hardware, camera, backend, FPS,
dropped frames, and CPU/GPU use in the validation report after a real run.

Regression tests must include normal driving, partial occlusion, low light, and false-alert
prevention. Mocked tests do not replace controlled/replay-camera testing on the reference setup.
Do not enable emotion classes before the evaluation gate passes. Usability acceptance requires
10 first-time non-technical participants, with at least 9/10 unaided temporary profile tasks and
8/10 correct interpretations of status, emotion/uncertainty, and alerts. Avoid collecting
participant photos or identifiers.

## Current validation limits

Automated unit and mocked integration tests do not run a physical camera, the proprietary
driver-behavior model, YOLO weights, DeepFace/TensorFlow inference, participant usability study,
or the licensed held-out evaluation data. These checks remain release prerequisites and must not
be reported as passed until performed and documented.
