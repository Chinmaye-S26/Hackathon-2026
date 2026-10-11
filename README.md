# SignalSync 🚦✋

**Traffic police hand signals that control the traffic lights, so the officer and the signal never disagree.**

Built for **Zero One Hackathon, Chapter 2** · Department of ISE, BGS College of Engineering and Technology, Bengaluru

---

## The problem

Bengaluru was the world's **2nd most congested city in 2025**, at **36 min 9 s per 10 km** and about **168 hours lost per year** in rush hour ([TomTom Traffic Index 2025](https://www.tomtom.com/traffic-index/bengaluru-traffic)). At busy junctions, traffic police still take over by hand. In July 2025 the Police Commissioner said *"Manual intervention is key"* ([Deccan Herald](https://www.deccanherald.com/amp/story/india%2Fkarnataka%2Fbengaluru%2Ftech-city-decides-to-bank-on-personnel-to-trim-traffic-woes-3627769)).

When an officer directs traffic by hand, the signal keeps running its own timer. Drivers see two conflicting instructions: the light says GREEN, the officer says STOP. To change the lights instead, the officer has to unlock the signal cabinet and switch to manual mode. That ties them to the box and leaves the signal stuck if they're called away ([US Patent 7,733,242](https://patents.google.com/patent/US7733242)).

## The solution

A camera watches the officer's hand. SignalSync recognises their signal and switches the traffic lights to match, with a safe amber and all-red transition in between. The officer keeps doing what they already do; the lights now follow.

| Gesture | Command | What the junction does |
|---|---|---|
| ✋ Open palm, held still | **ALL STOP** | Every road turns red (via amber) |
| ☝️ Point up | **North–South GO** | NS green, EW red |
| 👉 Point sideways (left or right) | **East–West GO** | EW green, NS red |
| ✌️ V sign | **Pedestrians WALK** | All vehicles red, WALK light on |
| ✊ Fist | **AUTO** | Back to the normal timer cycle |
| 👋 Wave side to side *(dynamic gesture)* | **CAUTION** | Flashing amber on all roads |

## Accident-proofing (why nothing triggers by mistake)

1. **Hold to confirm:** a static gesture must be held steadily for **1 second** before it fires. A progress bar shows the hold.
2. **Unclear = ignored:** diagonal points, pointing down, three fingers and half-bent hands are classed as *Unclear* and do nothing.
3. **Still palm only:** a moving open palm is not STOP. Only a still palm counts, and a wave is detected separately.
4. **No repeats:** a gesture can't fire again until the hand changes, and there's a 1 s cooldown between commands.
5. **Closest hand only:** small or far-away hands (bystanders) and low-confidence detections are ignored.
6. **Safety interlock:** the signal controller *always* goes green → amber → all-red → new green. Two greens, or WALK with a green, are impossible. This is checked in code and again on the Arduino.
7. **Fail-safe:** if the laptop stops talking to the Arduino for 3 s, the hardware signal falls back to flashing amber, just like real junction controllers.

Every command is also written to `logs/commands_YYYYMMDD.csv` with a timestamp. That gives an audit trail of who changed the junction and when.

## How it works

```
 Webcam ──frames──▶ MediaPipe Hand Landmarker ──21 landmarks──▶ Gesture classifier
                                                                     │ label per frame
                                                                     ▼
 Arduino LEDs ◀──"1000011"── Signal controller ◀──command── Hold / wave / safety filter
 On-screen junction ◀──lamp states──┘        └──▶ CSV audit log
```

1. **Capture:** OpenCV reads the webcam (640×480) and mirrors it.
2. **Detect:** the MediaPipe Hand Landmarker finds up to 2 hands and returns 21 landmarks each. SignalSync keeps the largest confident hand.
3. **Classify and confirm:** finger-extension rules give a label each frame. The wave detector and the 1 s hold filter decide whether a command fires. Unclear frames are dropped.
4. **Act:** the signal state machine changes the lights safely. The screen, the Arduino LEDs and the log are all updated.

**Measured on our Dell laptop (CPU only):** about 26 FPS and 30 ms per frame.

**Tech stack:** Python 3.9 to 3.12 · OpenCV · MediaPipe Tasks (Hand Landmarker, float16 model) · NumPy · pySerial · Arduino Uno/Nano (C++)

## Run it

```bash
git clone https://github.com/Chinmaye-S26/Hackathon-2026.git
cd Hackathon-2026
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python signalsync.py
```

The first run downloads the hand model (about 8 MB) into `models/`. If that download is blocked, get it from [this link](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) and save it as `models/hand_landmarker.task`.

| Option | Meaning |
|---|---|
| `--port COM3` | also drive the Arduino signal (Linux/macOS: `/dev/ttyUSB0`, `/dev/cu.usbmodem…`) |
| `--list-ports` | show the available serial ports |
| `--camera 1` | use another webcam |
| `--hold 1.0` | seconds a gesture must be held |
| `--benchmark` | guided accuracy test (see below) |

Keys: **Q** quits and **S** saves a screenshot to `screenshots/`.

## Hardware (optional, earns brownie points)

**Parts:** Arduino Uno/Nano, 2 red + 2 yellow + 2 green LEDs, 1 white/blue LED (pedestrian), 7 × 220 Ω resistors, breadboard, jumper wires.

| Arduino pin | LED | | Arduino pin | LED |
|---|---|---|---|---|
| 2 | NS Red | | 5 | EW Red |
| 3 | NS Yellow | | 6 | EW Yellow |
| 4 | NS Green | | 7 | EW Green |
| 8 | Pedestrian WALK | | GND | all LED short legs |

Wire each LED as *pin → 220 Ω → LED long leg*, with the short leg to GND. Upload `arduino/signalsync_arduino/signalsync_arduino.ino` with the Arduino IDE, close the Serial Monitor, then run `python signalsync.py --port COM3`. On power-up, all LEDs light for 0.5 s as a lamp test. The signals then flash amber until the laptop connects.

## Measure it (for the pitch)

```bash
python signalsync.py --benchmark --trials 5
```

The test asks for each of the 6 gestures 5 times in random order (30 tests). It then runs a 20-second "move naturally, no commands" period to count false triggers. It prints accuracy, per-gesture results, response time, processing time per frame and FPS, and saves everything to `results/benchmark_*.json`.

Logic tests that need no camera:

```bash
python tests/test_logic.py
```

## Project structure

```
signalsync.py        main app: camera loop, benchmark, logging
gestures.py          gesture classifier, wave detector, hold/confirm filter
controller.py        traffic signal state machine with safety interlocks
hardware.py          Arduino serial link with heartbeat
ui.py                on-screen dashboard
arduino/             Arduino sketch
tests/test_logic.py  classifier and signal-safety tests
```

## Team SynapX

| Name | USN | Semester |
|---|---|---|
| Chinmaye S | 1MP24IS015 | 5th |
| Lahari B | 1MP24IS026 | 5th |
| Adithi | 1MP25IS001 | 3rd |
| Prarthana | 1MP25IS033 | 3rd |

## Future scope

* Full-body arm signals (MediaPipe Pose) so officers can use standard arm signals from farther away
* Officer authentication: only a registered officer's hand or a reflective glove can take control
* Plug into existing junction controllers and the city's adaptive system (BATCS) as a "police override" input
* Emergency green corridor gesture for ambulances

## License

MIT
