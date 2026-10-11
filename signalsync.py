"""
SignalSync - traffic police hand signals that control the traffic lights.

Run:
    python signalsync.py                      # webcam + on-screen junction
    python signalsync.py --port COM3          # also drive the Arduino LED signals
    python signalsync.py --benchmark          # measure accuracy, response time, FPS
    python signalsync.py --list-ports         # find your Arduino's port
"""
import argparse
import csv
import json
import os
import random
import statistics
import time
import urllib.request
from datetime import datetime

import cv2

import ui
from controller import SignalController, MODE_TEXT
from gestures import (GesturePipeline, GESTURE_TO_COMMAND, GESTURE_NAMES, NO_HAND,
                      OPEN_PALM, POINT_UP, POINT_SIDE, V_SIGN, FIST, WAVE, palm_size)
from hardware import SerialLink, list_ports

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")


# --------------------------------------------------------------------------- detector
class HandDetector:
    """MediaPipe Hand Landmarker (Tasks API) in VIDEO mode."""

    def __init__(self, min_conf=0.7):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not os.path.exists(MODEL_PATH):
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            print("[setup] Downloading hand model (~8 MB, first run only)...")
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except Exception as e:
                raise SystemExit(f"Could not download the model: {e}\n"
                                 f"Download it manually from\n  {MODEL_URL}\nand save it as\n  {MODEL_PATH}")
        self.mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=0.6,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_ts = 0

    def detect(self, frame_bgr):
        """Returns a list of (landmarks_in_pixels, handedness_score)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = max(self.last_ts + 1, int(time.monotonic() * 1000))
        self.last_ts = ts
        result = self.landmarker.detect_for_video(image, ts)
        hands = []
        for i, lms in enumerate(result.hand_landmarks):
            pts = [(p.x * w, p.y * h) for p in lms]
            score = result.handedness[i][0].score if result.handedness else 1.0
            hands.append((pts, score))
        return hands

    def close(self):
        self.landmarker.close()


def pick_hand(hands, frame_h, min_hand=0.10, min_score=0.75):
    """Use only the largest (closest) confident hand, so bystanders in the
    background cannot control the junction."""
    best, best_size = None, 0.0
    for lm, score in hands:
        size = palm_size(lm)
        if score >= min_score and size >= min_hand * frame_h and size > best_size:
            best, best_size = lm, size
    return best


def open_camera(index):
    """Open the webcam, trying the backends that work on each OS
    (on Windows, DirectShow and Media Foundation; FFMPEG cannot open webcams)."""
    import sys
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    elif sys.platform == "darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok, _ = cap.read()
            if ok:
                print(f"[camera] Opened camera {index} ({cap.getBackendName()})")
                return cap
        cap.release()
    return None


# --------------------------------------------------------------------------- benchmark
class Benchmark:
    """Guided test: asks for each gesture N times in random order, then runs a
    'no command' period to count false triggers. Saves results/benchmark_*.json."""

    GESTURES = [OPEN_PALM, POINT_UP, POINT_SIDE, V_SIGN, FIST, WAVE]

    def __init__(self, trials, start):
        self.plan = []
        for _ in range(trials):
            order = self.GESTURES[:]
            random.shuffle(order)
            self.plan += order
        self.i = 0
        self.phase = "ready"
        self.phase_start = start
        self.first_seen = None
        self.results = []
        self.noise_triggers = 0
        self.noise_secs = 20.0
        self.done = False
        self.wrong = []

    def banner(self, now):
        if self.done:
            return ("Benchmark complete", "Results saved in the results folder. Press Q.")
        if self.phase == "noise":
            left = self.noise_secs - (now - self.phase_start)
            return ("NO COMMANDS: keep your hand moving", f"Talk with your hands, scratch, pick up a phone... {left:4.1f}s  (false: {self.noise_triggers})")
        g = self.plan[self.i]
        tag = f"Test {self.i + 1}/{len(self.plan)}"
        if self.phase == "ready":
            return (f"{tag}  NEXT: {GESTURE_NAMES[g].upper()}", "Drop your hand out of view, then wait...")
        return (f"{tag}  SHOW: {GESTURE_NAMES[g].upper()}  NOW", "Hold it until the green flash")

    def step(self, now, label, fired, pipeline):
        if self.done:
            return
        if self.phase == "ready":
            if now - self.phase_start >= 3.0:
                self.phase, self.phase_start, self.first_seen = "show", now, None
                self.wrong = []
                pipeline.reset()
                # A gesture still held over from the previous test must not count:
                # it only fires again after the hand changes shape.
                target = self.plan[self.i]
                if label not in (target, NO_HAND, "UNKNOWN", "PALM_MOVING"):
                    pipeline.confirm.block(label, now - pipeline.confirm.cooldown)
            return
        if self.phase == "noise":
            if fired:
                self.noise_triggers += 1
            if now - self.phase_start >= self.noise_secs:
                self.done = True
            return
        # phase == "show"
        target = self.plan[self.i]
        seen_label = OPEN_PALM if label == "PALM_MOVING" and target == WAVE else label
        if self.first_seen is None and seen_label == target:
            self.first_seen = now
        if fired and fired != target:
            self.wrong.append(fired)
        timeout = now - self.phase_start >= 6.0
        if fired == target or timeout:
            start = self.first_seen if self.first_seen is not None else self.phase_start
            ok = fired == target
            self.results.append({
                "target": target,
                "correct": ok,
                "wrong_fired": self.wrong[:],
                "response_ms": round((now - start) * 1000) if ok else None,
            })
            verdict = "OK" if ok else "MISSED"
            extra = f"  (also fired: {', '.join(GESTURE_NAMES[w] for w in self.wrong)})" if self.wrong else ""
            print(f"[test {self.i + 1:2d}] {GESTURE_NAMES[target]:15s} {verdict}{extra}")
            self.i += 1
            if self.i >= len(self.plan):
                self.phase, self.phase_start = "noise", now
                pipeline.reset()
            else:
                self.phase, self.phase_start = "ready", now

    def save(self, frame_ms, fps_list, hold):
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        n = len(self.results)
        correct = sum(r["correct"] for r in self.results)
        resp = [r["response_ms"] for r in self.results if r["correct"]]
        per = {}
        for g in self.GESTURES:
            rs = [r for r in self.results if r["target"] == g]
            if rs:
                per[GESTURE_NAMES[g]] = f"{sum(r['correct'] for r in rs)}/{len(rs)}"
        summary = {
            "tests": n,
            "accuracy_pct": round(100 * correct / n, 1) if n else None,
            "per_gesture": per,
            "tests_with_a_wrong_command": sum(1 for r in self.results if r["wrong_fired"]),
            "missed": sum(1 for r in self.results if not r["correct"]),
            "avg_response_ms_incl_hold": round(statistics.mean(resp)) if resp else None,
            "hold_time_s": hold,
            "avg_processing_ms_per_frame": round(statistics.mean(frame_ms), 1) if frame_ms else None,
            "avg_fps": round(statistics.mean(fps_list), 1) if fps_list else None,
            "false_triggers_in_noise": self.noise_triggers,
            "noise_period_s": self.noise_secs,
        }
        path = os.path.join(HERE, "results", f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(path, "w") as f:
            json.dump({"summary": summary, "trials": self.results}, f, indent=2)
        print("\n=========== SignalSync benchmark ===========")
        for k, v in summary.items():
            print(f"  {k:30s} {v}")
        print(f"  saved to {path}\n")


# --------------------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description="SignalSync - gesture-controlled traffic signals")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--port", help="Arduino serial port, e.g. COM3 or /dev/ttyUSB0")
    ap.add_argument("--hold", type=float, default=1.0, help="seconds a gesture must be held (default 1.0)")
    ap.add_argument("--green", type=float, default=8.0, help="AUTO mode green time in seconds")
    ap.add_argument("--min-hand", type=float, default=0.10,
                    help="ignore hands smaller than this fraction of frame height (default 0.10)")
    ap.add_argument("--benchmark", action="store_true", help="run the guided accuracy test")
    ap.add_argument("--trials", type=int, default=5, help="benchmark rounds per gesture (default 5)")
    ap.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    args = ap.parse_args()

    if args.list_ports:
        ports = list_ports()
        print("\n".join(ports) if ports else "No serial ports found.")
        return

    cap = open_camera(args.camera)
    if cap is None:
        raise SystemExit(
            f"Cannot open camera {args.camera}.\n"
            "  1. Close any app using the camera (Zoom, Teams, Camera app, browser tabs).\n"
            "  2. Windows Settings > Privacy & security > Camera: turn ON 'Camera access'\n"
            "     and 'Let desktop apps access your camera'.\n"
            "  3. Check the laptop's camera shutter/privacy switch or the F-key with a camera icon.\n"
            "  4. Try another index: python signalsync.py --camera 1")

    detector = HandDetector()
    pipeline = GesturePipeline(hold=args.hold)
    now = time.monotonic()
    signals = SignalController(now=now, green=args.green)
    link = SerialLink(args.port)
    bench = Benchmark(args.trials, now) if args.benchmark else None

    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "screenshots"), exist_ok=True)
    log_path = os.path.join(HERE, "logs", f"commands_{datetime.now():%Y%m%d}.csv")
    new_file = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log_csv = csv.writer(log_file)
    if new_file:
        log_csv.writerow(["timestamp", "gesture", "command", "mode_after"])

    screen_log = []
    flash_text, flash_until = None, 0.0
    frame_ms, fps_list = [], []
    last_frame = time.monotonic()
    fps = 0.0

    cv2.namedWindow("SignalSync", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SignalSync", ui.W, ui.H)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame not received.")
                break
            frame = cv2.flip(frame, 1)                     # mirror, like a selfie view
            fh = frame.shape[0]

            t0 = time.perf_counter()
            hands = detector.detect(frame)
            lm = pick_hand(hands, fh, args.min_hand)
            now = time.monotonic()
            label, fired = pipeline.step(now, lm)
            proc = (time.perf_counter() - t0) * 1000
            frame_ms.append(proc)

            dt = now - last_frame
            last_frame = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
                fps_list.append(fps)

            if fired:
                command = GESTURE_TO_COMMAND[fired]
                changed = signals.request(command, now)
                stamp = datetime.now().strftime("%H:%M:%S")
                note = "" if changed else " (already)"
                screen_log.append(f"{stamp}  {GESTURE_NAMES[fired]} -> {ui.COMMAND_TEXT[command]}{note}")
                log_csv.writerow([datetime.now().isoformat(timespec="seconds"),
                                  fired, command, signals.mode])
                log_file.flush()
                flash_text, flash_until = ui.COMMAND_TEXT[command], now + 0.8

            if bench:
                bench.step(now, label, fired, pipeline)

            lamps = signals.update(now)
            link.send(lamps, now)

            # scale landmarks to the displayed camera size
            disp_lm = None
            if lm is not None:
                sx, sy = ui.CAM_W / frame.shape[1], ui.CAM_H / fh
                disp_lm = [(x * sx, y * sy) for x, y in lm]
            progress = pipeline.confirm.progress(now)
            hand_state = "fire" if now < flash_until else ("hold" if progress > 0 else "idle")

            canvas = ui.render(
                frame, lamps=lamps, mode=signals.mode, phase=signals.phase_text(),
                label=label, progress=progress, hand_lm=disp_lm, hand_state=hand_state,
                fps=fps, latency_ms=proc, log=screen_log, hardware=link.connected,
                banner=bench.banner(now) if bench else None,
                flash=flash_text if now < flash_until else None,
            )
            cv2.imshow("SignalSync", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = os.path.join(HERE, "screenshots", f"signalsync_{datetime.now():%Y%m%d_%H%M%S}.png")
                cv2.imwrite(path, canvas)
                print(f"Saved {path}")
    except KeyboardInterrupt:
        pass  # Ctrl+C is a normal way to stop
    finally:
        if bench and bench.results:
            bench.save(frame_ms[30:], fps_list[30:], args.hold)   # skip warm-up frames
        elif frame_ms:
            print(f"Average processing: {statistics.mean(frame_ms[30:] or frame_ms):.1f} ms/frame, "
                  f"average {statistics.mean(fps_list[30:] or fps_list):.1f} FPS")
        log_file.close()
        link.close()
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
