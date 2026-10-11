"""Logic tests that need no camera: python -m pytest tests  (or python tests/test_logic.py)"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gestures import (classify, GesturePipeline, OPEN_PALM, POINT_UP, POINT_SIDE,
                      V_SIGN, FIST, WAVE, UNKNOWN)
from controller import SignalController, AUTO, GO_NS, GO_EW, ALL_STOP, PEDESTRIAN, CAUTION, G, R

FINGER_X = {"index": -36, "middle": -12, "ring": 12, "pinky": 34}
BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def make_hand(up, angle_deg=0.0, origin=(320, 400), scale=1.0):
    """Synthetic 21-point hand pointing up (before rotation). `up` = extended fingers."""
    pts = [None] * 21
    pts[0] = (0, 0)
    pts[1], pts[2], pts[3], pts[4] = (-30, -20), (-50, -45), (-62, -65), (-70, -85)   # thumb
    for name, b in BASE.items():
        x = FINGER_X[name]
        if name in up:
            ys = [-90, -130, -155, -180]
        else:                                   # curled back toward the palm
            ys = [-90, -110, -85, -70]
        for k in range(4):
            pts[b + k] = (x, ys[k])
    a = math.radians(angle_deg)
    out = []
    for x, y in pts:
        rx = x * math.cos(a) - y * math.sin(a)
        ry = x * math.sin(a) + y * math.cos(a)
        out.append((origin[0] + rx * scale, origin[1] + ry * scale))
    return out


ALL4 = ["index", "middle", "ring", "pinky"]


def test_static_classification():
    assert classify(make_hand(ALL4)) == OPEN_PALM
    assert classify(make_hand(ALL4, 40)) == OPEN_PALM
    assert classify(make_hand([])) == FIST
    assert classify(make_hand(["index", "middle"])) == V_SIGN
    assert classify(make_hand(["index"])) == POINT_UP
    assert classify(make_hand(["index"], 90)) == POINT_SIDE     # pointing right
    assert classify(make_hand(["index"], -90)) == POINT_SIDE    # pointing left
    assert classify(make_hand(["index"], 45)) == UNKNOWN        # diagonal ignored
    assert classify(make_hand(["index"], 180)) == UNKNOWN       # pointing down ignored
    assert classify(make_hand(["index", "middle", "ring"])) == UNKNOWN


def run(pipe, frames, t0=0.0, fps=30):
    fired = []
    for i, lm in enumerate(frames):
        _, f = pipe.step(t0 + i / fps, lm)
        if f:
            fired.append((f, round(i / fps, 2)))
    return fired


def test_hold_required_and_no_repeat():
    pipe = GesturePipeline(hold=1.0)
    short = [make_hand(["index"])] * 20                     # 0.67 s: too short
    assert run(pipe, short) == []
    pipe = GesturePipeline(hold=1.0)
    fired = run(pipe, [make_hand(["index"])] * 120)          # 4 s held
    assert [f for f, _ in fired] == [POINT_UP]               # fires once only


def test_flicker_tolerated_but_noise_ignored():
    pipe = GesturePipeline(hold=1.0)
    frames = []
    for i in range(60):
        frames.append(make_hand([]) if i % 15 == 7 else make_hand(["index", "middle"]))
    assert [f for f, _ in run(pipe, frames)] == [V_SIGN]
    pipe = GesturePipeline(hold=1.0)
    random.seed(1)
    shapes = [ALL4, [], ["index"], ["index", "middle"], ["index", "middle", "ring"]]
    noise = [make_hand(random.choice(shapes)) for _ in range(300)]  # random shapes every frame
    assert run(pipe, noise) == []


def test_wave_and_moving_palm():
    pipe = GesturePipeline(hold=1.0)
    frames = []
    for i in range(60):                                   # 2 s of waving, ~1.5 Hz
        x = 320 + 90 * math.sin(2 * math.pi * 1.5 * i / 30)
        frames.append(make_hand(ALL4, origin=(x, 400)))
    names = [f for f, _ in run(pipe, frames)]
    assert names == [WAVE], names                          # wave, and NOT also stop
    pipe = GesturePipeline(hold=1.0)
    still = [make_hand(ALL4)] * 45
    assert [f for f, _ in run(pipe, still)] == [OPEN_PALM]


def test_signal_safety_random():
    random.seed(7)
    modes = [AUTO, GO_NS, GO_EW, ALL_STOP, PEDESTRIAN, CAUTION]
    ctl = SignalController(now=0.0)
    t = 0.0
    for _ in range(3000):
        t += random.choice([0.05, 0.3, 1.0, 2.5])
        if random.random() < 0.3:
            ctl.request(random.choice(modes), t)
        lamps = ctl.update(t)
        assert lamps.is_safe(), lamps


def test_green_goes_through_amber_and_all_red():
    ctl = SignalController(now=0.0)
    ctl.request(GO_EW, 0.0)
    seen = []
    t = 0.0
    while t < 6:
        l = ctl.update(t)
        if not seen or seen[-1] != (l.ns, l.ew):
            seen.append((l.ns, l.ew))
        t += 0.1
    assert seen == [("Y", "R"), ("R", "R"), ("R", "G")], seen


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
