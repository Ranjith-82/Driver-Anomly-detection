from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import math
import queue
import struct
import subprocess
import threading
import time
import wave


FATIGUE_ALERT_COOLDOWN = 45.0
FATIGUE_BEEP_COOLDOWN = 10.0
FATIGUE_BEEP_FREQUENCY = 1000
FATIGUE_BEEP_DURATION = 350
DISTRACTION_ALERT_DELAY = 1.5
DISTRACTION_BEEP_INTERVAL = 4.0
DISTRACTION_VOICE_INTERVAL = 4.0
YAWN_ALERT_COOLDOWN = 10.0
YAWN_BEEP_DURATION = 120
YAWN_BEEP_FREQUENCY = 650
DISTRACTION_BEEP_DURATION = 180
DISTRACTION_BEEP_FREQUENCY = 900
FATIGUE_MESSAGE = (
    "You may be feeling drowsy. Please have a cup of coffee."
)
DISTRACTION_MESSAGE = "You are distracted. Please be focused."
TALKING_ALERT_COOLDOWN = 120.0
TALKING_ALERT_THRESHOLD = 4
SAMPLE_RATE = 44100
OBJECT_ALERTS = {
    0: (
        "object_phone",
        "Possible phone use near the identified driver. "
        "Please keep the phone away while driving.",
    ),
    1: (
        "object_cigarette",
        "Cigarette detected. Please focus on driving.",
    ),
    2: ("object_cup", "Cup detected. Please focus on driving."),
    # 3: ("object_bottle", "Bottle detected. Please focus on driving."),
}


@dataclass(frozen=True)
class AudioEvent:
    kind: str
    key: str
    payload: object
    cancel_when_inactive: bool = True


def _tone_wave(
    frequency: int,
    duration_ms: int,
    *,
    count: int = 1,
    gap_ms: int = 120,
) -> bytes:
    samples = bytearray()
    tone_length = round(SAMPLE_RATE * duration_ms / 1000)
    silence_length = round(SAMPLE_RATE * gap_ms / 1000)
    fade_length = max(1, round(SAMPLE_RATE * 0.008))
    for tone_index in range(count):
        for sample_index in range(tone_length):
            envelope = min(
                1.0,
                sample_index / fade_length,
                (tone_length - sample_index - 1) / fade_length,
            )
            value = int(
                14000
                * max(0.0, envelope)
                * math.sin(2.0 * math.pi * frequency * sample_index / SAMPLE_RATE)
            )
            samples.extend(struct.pack("<h", value))
        if tone_index + 1 < count:
            samples.extend(b"\x00\x00" * silence_length)

    output = io.BytesIO()
    with wave.open(output, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(SAMPLE_RATE)
        audio_file.writeframes(samples)
    return output.getvalue()


class WindowsSpeechEngine:
    """Keeps one local Windows speech synthesizer alive for the audio worker."""

    _SCRIPT = """
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
[Console]::Out.WriteLine('READY')
while ($true) {
    $line = [Console]::In.ReadLine()
    if ($null -eq $line -or $line -eq 'EXIT') { break }
    try {
        $text = [Text.Encoding]::UTF8.GetString(
            [Convert]::FromBase64String($line)
        )
        $speaker.Speak($text)
        [Console]::Out.WriteLine('OK')
    } catch {
        [Console]::Out.WriteLine('ERROR:' + $_.Exception.Message)
    }
}
$speaker.Dispose()
"""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None

    def start(self) -> bool:
        encoded_script = base64.b64encode(
            self._SCRIPT.encode("utf-16le")
        ).decode("ascii")
        try:
            self._process = subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-EncodedCommand",
                    encoded_script,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as error:
            print(f"[VOICE] Could not start Windows speech engine: {error}")
            return False

        if self._process.stdout is None:
            print("[VOICE] Speech engine has no output channel.")
            self.close()
            return False
        response = self._process.stdout.readline().strip()
        if response != "READY":
            detail = response or "PowerShell exited before engine initialization."
            process = self._process
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stderr is not None:
                error_output = process.stderr.read().strip()
                if error_output:
                    detail = f"{detail}\n{error_output}"
            print(f"[VOICE] Speech engine initialization failed: {detail}")
            self.close()
            return False
        print("[VOICE] Windows speech engine initialized.")
        return True

    def speak(self, message: str) -> bool:
        process = self._process
        if (
            process is None
            or process.poll() is not None
            or process.stdin is None
            or process.stdout is None
        ):
            print("[VOICE] Speech engine is not available.")
            return False
        encoded_message = base64.b64encode(
            message.encode("utf-8")
        ).decode("ascii")
        try:
            process.stdin.write(encoded_message + "\n")
            process.stdin.flush()
            response = process.stdout.readline().strip()
        except (BrokenPipeError, OSError) as error:
            print(f"[VOICE] Speech playback failed: {error}")
            return False
        if response == "OK":
            print(f"[VOICE] Spoken: {message}")
            return True
        print(
            f"[VOICE] Speech playback failed: {response or 'engine stopped'}")
        return False

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None and process.poll() is None:
                process.stdin.write("EXIT\n")
                process.stdin.flush()
                process.stdin.close()
            process.wait(timeout=2.0)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired) as error:
            print(f"[VOICE] Speech engine did not shut down cleanly: {error}")
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()


class AlertAudioWorker(threading.Thread):
    """Plays audio independently of camera capture and dashboard rendering."""

    def __init__(self, *, muted: bool = False) -> None:
        super().__init__(name="alert-audio-worker", daemon=True)
        self.muted = muted
        self._queue: queue.Queue[AudioEvent] = queue.Queue(maxsize=16)
        self._pending: set[tuple[str, str]] = set()
        self._lock = threading.Lock()
        self._active_keys: set[str] = set()
        self._stop_event = threading.Event()
        self._winsound = None

    def run(self) -> None:
        if self.muted:
            print("[AUDIO] Audio disabled by --mute.")
            return
        try:
            import winsound
        except ImportError as error:
            print(
                "[AUDIO] winsound is unavailable; Windows audio playback "
                f"cannot start: {error}"
            )
        else:
            self._winsound = winsound
            print("[AUDIO] Windows waveform playback ready.")

        speech = WindowsSpeechEngine()
        speech_ready = speech.start()
        if not speech_ready:
            print("[AUDIO] Beep playback remains available; voice is disabled.")

        print("[AUDIO] Alert audio worker started.")
        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    event = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                with self._lock:
                    self._pending.discard((event.key, event.kind))
                    active = event.key in self._active_keys
                try:
                    if event.cancel_when_inactive and not active:
                        print(
                            f"[AUDIO] Dropped obsolete {event.kind} for "
                            f"{event.key}."
                        )
                        continue
                    print(
                        f"[AUDIO] Processing {event.kind} for {event.key}."
                    )
                    if event.kind == "voice":
                        if speech_ready:
                            speech.speak(str(event.payload))
                        else:
                            print(
                                f"[VOICE] Not spoken (speech engine "
                                f"unavailable): {event.payload}"
                            )
                    elif event.kind == "beep":
                        frequency, duration_ms, count, gap_ms = event.payload
                        self._play_beeps(
                            frequency, duration_ms, count, gap_ms
                        )
                except Exception as error:
                    print(
                        f"[AUDIO] Playback failed for {event.key} "
                        f"({event.kind}): {error}"
                    )
                finally:
                    self._queue.task_done()
        finally:
            speech.close()
            print("[AUDIO] Alert audio worker stopped.")

    def set_active_alerts(self, active_keys: set[str]) -> None:
        with self._lock:
            self._active_keys = set(active_keys)

    def enqueue_voice(
        self,
        key: str,
        message: str,
        *,
        cancel_when_inactive: bool = True,
    ) -> bool:
        return self._enqueue(
            AudioEvent("voice", key, message, cancel_when_inactive)
        )

    def enqueue_beeps(
        self,
        key: str,
        frequency: int,
        duration_ms: int,
        *,
        count: int = 1,
        gap_ms: int = 120,
        cancel_when_inactive: bool = True,
    ) -> bool:
        return self._enqueue(
            AudioEvent(
                "beep",
                key,
                (frequency, duration_ms, count, gap_ms),
                cancel_when_inactive,
            )
        )

    def _enqueue(self, event: AudioEvent) -> bool:
        if self.muted:
            print(f"[AUDIO] Audio disabled by --mute; skipped {event.kind}.")
            return False
        dedupe_key = (event.key, event.kind)
        with self._lock:
            if dedupe_key in self._pending:
                print(
                    f"[AUDIO] Skipped duplicate queued {event.kind} for "
                    f"{event.key}."
                )
                return False
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                print(
                    f"[AUDIO] Queue full; could not queue {event.kind} "
                    f"for {event.key}."
                )
                return False
            self._pending.add(dedupe_key)
        print(f"[AUDIO] Queued {event.kind} for {event.key}.")
        return True

    def _play_beeps(
        self, frequency: int, duration_ms: int, count: int, gap_ms: int
    ) -> bool:
        if self._winsound is None:
            print(
                "[BEEP] Not played: Windows winsound playback is "
                "unavailable."
            )
            return False
        audio = _tone_wave(
            frequency, duration_ms, count=count, gap_ms=gap_ms
        )
        try:
            self._winsound.PlaySound(
                audio, self._winsound.SND_MEMORY
            )
        except RuntimeError as error:
            print(f"[BEEP] Wave playback failed; trying system beep: {error}")
            try:
                self._winsound.Beep(frequency, duration_ms)
                if count == 2:
                    time.sleep(gap_ms / 1000)
                    self._winsound.Beep(frequency, duration_ms)
            except RuntimeError as fallback_error:
                print(f"[BEEP] System beep fallback failed: {fallback_error}")
                return False
        print(
            f"[BEEP] Played {count} x {frequency} Hz, "
            f"{duration_ms} ms."
        )
        return True

    def close(self, timeout: float = 3.0) -> None:
        self.set_active_alerts(set())
        self._stop_event.set()
        with self._lock:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()
            self._pending.clear()
        if self.ident is not None:
            self.join(timeout=timeout)


class AlertAudioManager:
    """Schedules audio from detector states without changing detection logic."""

    def __init__(
        self,
        worker: AlertAudioWorker,
        *,
        enabled: bool = True,
        talking_alert_threshold: int = TALKING_ALERT_THRESHOLD,
        fatigue_seconds: float = 1.5,
    ) -> None:
        self.worker = worker
        self.enabled = enabled
        self.talking_alert_threshold = talking_alert_threshold
        self.fatigue_seconds = fatigue_seconds
        self.distraction_started: float | None = None
        self.last_fatigue_voice = -float("inf")
        self.last_fatigue_beep = -float("inf")
        self.last_distraction_beep = -float("inf")
        self.last_distraction_voice = -float("inf")
        self.last_yawn_alert = -float("inf")
        self.last_talking_voice = -float("inf")
        self.talking_episodes = 0
        self.was_talking = False
        self.previous_behavior: str | None = None
        self.distraction_condition_active = False
        self.active_object_classes: set[int] = set()

    def update(self, values: dict[str, object], now: float) -> float:
        behavior = str(values.get("behavior", "UNKNOWN"))
        if behavior != self.previous_behavior:
            print(f"[ALERT AUDIO] behavior={behavior!r}")
            self.previous_behavior = behavior

        face_present = values.get("face_present") is True
        closed_duration = values.get("eyes_closed_duration")
        eye_closure_fatigue = (
            isinstance(closed_duration, (int, float))
            and not isinstance(closed_duration, bool)
            and math.isfinite(float(closed_duration))
            and closed_duration >= self.fatigue_seconds
        )
        fatigue = face_present and (
            behavior == "FATIGUE ALERT"
            or values.get("fatigue_alert") is True
            or eye_closure_fatigue
        )
        active_keys: set[str] = set()
        if fatigue:
            active_keys.add("fatigue")

        distraction_detected = not fatigue and face_present and (
            behavior == "DISTRACTED"
            or values.get("gaze_distraction_confirmed") is True
        )
        if distraction_detected:
            if self.distraction_started is None:
                self.distraction_started = now
            distraction_duration = max(0.0, now - self.distraction_started)
            if distraction_duration >= DISTRACTION_ALERT_DELAY:
                active_keys.add("distraction")
        else:
            distraction_duration = 0.0
            self.distraction_started = None
            self.last_distraction_beep = -float("inf")
            self.last_distraction_voice = -float("inf")

        distraction_condition_met = "distraction" in active_keys
        if distraction_condition_met and not self.distraction_condition_active:
            print(
                f"[ALERT AUDIO] Distraction condition met; "
                f"duration={distraction_duration:.2f}s."
            )
        self.distraction_condition_active = distraction_condition_met

        self.worker.set_active_alerts(active_keys)

        if fatigue and now - self.last_fatigue_beep >= FATIGUE_BEEP_COOLDOWN:
            self._beep(
                "fatigue",
                FATIGUE_BEEP_FREQUENCY,
                FATIGUE_BEEP_DURATION,
                cancel_when_inactive=False,
            )
            self.last_fatigue_beep = now

        if fatigue and now - self.last_fatigue_voice >= FATIGUE_ALERT_COOLDOWN:
            self._voice("fatigue", FATIGUE_MESSAGE)
            self.last_fatigue_voice = now
            print("[ALERT AUDIO] Fatigue condition met.")

        if distraction_condition_met:
            if now - self.last_distraction_beep >= DISTRACTION_BEEP_INTERVAL:
                self._beep(
                    "distraction",
                    DISTRACTION_BEEP_FREQUENCY,
                    DISTRACTION_BEEP_DURATION,
                )
                self.last_distraction_beep = now
            if (
                now - self.last_distraction_voice
                >= DISTRACTION_VOICE_INTERVAL
            ):
                self._voice("distraction", DISTRACTION_MESSAGE)
                self.last_distraction_voice = now

        if values.get("yawn_event") is True:
            print("[ALERT AUDIO] Yawn event condition met.")
            if now - self.last_yawn_alert >= YAWN_ALERT_COOLDOWN:
                self._beep(
                    "yawn",
                    YAWN_BEEP_FREQUENCY,
                    YAWN_BEEP_DURATION,
                    count=2,
                    gap_ms=120,
                    cancel_when_inactive=False,
                )
                self.last_yawn_alert = now

        detections = values.get("object_detections")
        object_classes: set[int] = set()
        if isinstance(detections, list):
            object_classes = {
                detection["class_id"]
                for detection in detections
                if isinstance(detection, dict)
                and detection.get("class_id") in OBJECT_ALERTS
                and detection.get("class_id") != 0
            }
        if (
            values.get("driver_phone_alert") is True
            and values.get("driver_confirmed") is True
            and values.get("phone_detection_confirmed") is True
        ):
            object_classes.add(0)
        for class_id in sorted(object_classes - self.active_object_classes):
            key, message = OBJECT_ALERTS[class_id]
            print(f"[ALERT AUDIO] New object alert condition met: {key}.")
            self._beep(
                key,
                DISTRACTION_BEEP_FREQUENCY,
                DISTRACTION_BEEP_DURATION,
                cancel_when_inactive=False,
            )
            self._voice(
                key, message, cancel_when_inactive=False
            )
        self.active_object_classes = object_classes

        is_talking = behavior == "TALKING" and face_present
        if is_talking and not self.was_talking:
            self.talking_episodes += 1
            print(
                f"[ALERT AUDIO] Talking episode "
                f"{self.talking_episodes}/{self.talking_alert_threshold}."
            )
            if (
                self.talking_episodes >= self.talking_alert_threshold
                and now - self.last_talking_voice
                >= TALKING_ALERT_COOLDOWN
            ):
                self._voice(
                    "talking",
                    "Repeated talking detected. Please focus on driving "
                    "and avoid conversation while the vehicle is moving.",
                    cancel_when_inactive=False,
                )
                self.last_talking_voice = now
        self.was_talking = is_talking

        return distraction_duration

    def _voice(
        self,
        key: str,
        message: str,
        *,
        cancel_when_inactive: bool = True,
    ) -> None:
        if not self.enabled:
            print(f"[AUDIO] Audio disabled by --mute; skipped {key} voice.")
            return
        print(f"[ALERT AUDIO] {key} voice condition met.")
        self.worker.enqueue_voice(
            key, message, cancel_when_inactive=cancel_when_inactive
        )

    def _beep(
        self,
        key: str,
        frequency: int,
        duration_ms: int,
        *,
        count: int = 1,
        gap_ms: int = 120,
        cancel_when_inactive: bool = True,
    ) -> None:
        if not self.enabled:
            print(f"[AUDIO] Audio disabled by --mute; skipped {key} beep.")
            return
        self.worker.enqueue_beeps(
            key,
            frequency,
            duration_ms,
            count=count,
            gap_ms=gap_ms,
            cancel_when_inactive=cancel_when_inactive,
        )
