"""
Gesture recognition for SignalSync.

Takes the 21 hand landmarks from MediaPipe and turns them into one of six
commands. Three layers keep it accident-proof:

  1. classify()        - per-frame label, or UNKNOWN if the hand shape is unclear
  2. WaveDetector      - spots the dynamic side-to-side wave, and tells whether
                         an open palm is held STILL (a moving palm is not a command)
  3. GestureConfirmer  - a static gesture must be held steadily for HOLD seconds
                         before it fires, and cannot re-fire until the hand changes
"""
import math
from collections import deque

# Landmark indices (MediaPipe hand model)
WRIST = 0
FINGERS = {             # (MCP, PIP, TIP)
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}

# Gesture labels
OPEN_PALM = "OPEN_PALM"
POINT_UP = "POINT_UP"
POINT_SIDE = "POINT_SIDE"
V_SIGN = "V_SIGN"
FIST = "FIST"
WAVE = "WAVE"
PALM_MOVING = "PALM_MOVING"   # open palm that is not held still -> ignored
UNKNOWN = "UNKNOWN"
NO_HAND = "NO_HAND"

STATIC_COMMAND_GESTURES = {OPEN_PALM, POINT_UP, POINT_SIDE, V_SIGN, FIST}

# What each gesture means at the junction
GESTURE_TO_COMMAND = {
    OPEN_PALM: "ALL_STOP",
    POINT_UP: "GO_NS",
    POINT_SIDE: "GO_EW",
    V_SIGN: "PEDESTRIAN",
    FIST: "AUTO",
    WAVE: "CAUTION",
}

GESTURE_NAMES = {
    OPEN_PALM: "Open palm",
    POINT_UP: "Point up",
    POINT_SIDE: "Point sideways",
    V_SIGN: "V sign",
    FIST: "Fist",
    WAVE: "Wave",
    PALM_MOVING: "Palm (moving)",
    UNKNOWN: "Unclear",
    NO_HAND: "No hand",
}


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def palm_size(lm):
    """Wrist to middle-finger knuckle, in pixels. Used as the hand's scale."""
    return _dist(lm[WRIST], lm[9])


def finger_states(lm):
    """Which of the four fingers are extended. Rotation independent:
    an extended finger's tip is clearly farther from the wrist than its middle joint."""
    w = lm[WRIST]
    states = {}
    for name, (mcp, pip, tip) in FINGERS.items():
        d_tip, d_pip, d_mcp = _dist(w, lm[tip]), _dist(w, lm[pip]), _dist(w, lm[mcp])
        states[name] = d_tip > d_pip * 1.12 and d_tip > d_mcp * 1.35
    return states


def classify(lm):
    """lm: list of 21 (x, y) points in PIXELS (so the aspect ratio is correct).
    Returns one gesture label. Anything ambiguous returns UNKNOWN."""
    f = finger_states(lm)
    up = [n for n in ("index", "middle", "ring", "pinky") if f[n]]

    if len(up) == 4:
        return OPEN_PALM
    if len(up) == 0:
        return FIST
    if up == ["index", "middle"]:
        return V_SIGN
    if up == ["index"]:
        dx = lm[8][0] - lm[5][0]
        dy = lm[8][1] - lm[5][1]           # image y grows downwards
        if dy < 0 and abs(dy) > 1.3 * abs(dx):
            return POINT_UP
        if abs(dx) > 1.3 * abs(dy):
            return POINT_SIDE
        return UNKNOWN                     # diagonal or pointing down: ignored
    return UNKNOWN


class WaveDetector:
    """Detects a side-to-side wave of an open palm (a dynamic gesture) and
    reports whether an open palm is being held still."""

    def __init__(self, window=2.5, min_reversals=2, amplitude=0.6, still_limit=0.30,
                 min_palm_ratio=0.5, lost_timeout=0.5):
        self.window = window              # seconds of motion looked at
        self.min_reversals = min_reversals  # 2 = left-right-left
        self.amplitude = amplitude        # min swing, in palm sizes
        self.still_limit = still_limit    # max drift (palm sizes) in 0.5 s to count as still
        self.min_palm_ratio = min_palm_ratio  # share of frames that must be an open palm
        self.lost_timeout = lost_timeout  # forget the motion if the hand is gone this long
        self.track = deque()              # (t, x in palm units, is_open_palm)
        self.last_seen = -1e9

    def reset(self):
        self.track.clear()

    def update(self, t, label, lm):
        """Returns (is_wave, is_still).
        A fast-moving hand is often blurred, so a few frames that are not
        classed as an open palm (or briefly lost) do not break the wave."""
        if lm is None:
            if t - self.last_seen > self.lost_timeout:
                self.track.clear()
            return False, False
        self.last_seen = t
        x = lm[WRIST][0] / max(palm_size(lm), 1e-6)
        self.track.append((t, x, label == OPEN_PALM))
        while self.track and t - self.track[0][0] > self.window:
            self.track.popleft()

        recent = [p for tt, p, palm in self.track if t - tt <= 0.5 and palm]
        still = (label == OPEN_PALM and len(recent) >= 3
                 and (max(recent) - min(recent)) < self.still_limit)

        palm_ratio = sum(1 for *_, palm in self.track if palm) / len(self.track)
        if palm_ratio < self.min_palm_ratio:
            return False, still

        # count direction reversals, with hysteresis so jitter does not count
        reversals, direction = 0, 0
        anchor = self.track[0][1]
        for _, p, _ in self.track:
            if direction >= 0 and p < anchor - self.amplitude:
                if direction == 1:
                    reversals += 1
                direction, anchor = -1, p
            elif direction <= 0 and p > anchor + self.amplitude:
                if direction == -1:
                    reversals += 1
                direction, anchor = 1, p
            elif (direction == 1 and p > anchor) or (direction == -1 and p < anchor):
                anchor = p                # follow the extreme point
        return reversals >= self.min_reversals, still


class GestureConfirmer:
    """A static gesture must be held for `hold` seconds (a few flickering frames
    are tolerated) before it fires. After firing, the same gesture cannot fire
    again until the hand shows something else, and no command fires within
    `cooldown` seconds of the previous one."""

    def __init__(self, hold=1.0, tolerance_frames=3, cooldown=1.0):
        self.hold = hold
        self.tolerance = tolerance_frames
        self.cooldown = cooldown
        self.reset()

    def reset(self):
        self.run_label = NO_HAND
        self.run_start = 0.0
        self.bad = 0
        self.blocked = None
        self.last_fire = -1e9

    def block(self, label, t):
        """Stop `label` from firing (used after a wave so the palm does not also fire STOP)."""
        self.blocked = label
        self.run_label, self.run_start, self.bad = label, t, 0
        self.last_fire = t

    def progress(self, t):
        if self.run_label not in STATIC_COMMAND_GESTURES or self.run_label == self.blocked:
            return 0.0
        return max(0.0, min(1.0, (t - self.run_start) / self.hold))

    def update(self, t, label):
        """Feed one frame's label. Returns the gesture that fired, or None."""
        if label == self.run_label:
            self.bad = 0
        else:
            self.bad += 1
            if self.bad > self.tolerance:
                self.run_label, self.run_start, self.bad = label, t, 0
                same_palm = self.blocked == OPEN_PALM and label == PALM_MOVING
                if label != self.blocked and not same_palm:
                    self.blocked = None   # hand changed: re-arm

        if (self.run_label in STATIC_COMMAND_GESTURES
                and self.run_label != self.blocked
                and t - self.run_start >= self.hold
                and t - self.last_fire >= self.cooldown):
            self.blocked = self.run_label
            self.last_fire = t
            return self.run_label
        return None


class GesturePipeline:
    """Puts the three layers together. Call step() once per frame."""

    def __init__(self, hold=1.0, cooldown=1.0):
        self.wave = WaveDetector()
        self.confirm = GestureConfirmer(hold=hold, cooldown=cooldown)
        self.wave_cooldown = 2.0
        self.last_wave = -1e9

    def reset(self):
        self.wave.reset()
        self.confirm.reset()

    def step(self, t, lm):
        """lm: 21 pixel landmarks of the chosen hand, or None.
        Returns (frame_label, fired_gesture_or_None)."""
        if lm is None:
            label = NO_HAND
        else:
            label = classify(lm)

        is_wave, still = self.wave.update(t, label, lm)
        if is_wave and t - self.last_wave >= self.wave_cooldown:
            self.last_wave = t
            self.wave.reset()
            self.confirm.block(OPEN_PALM, t)
            return WAVE, WAVE

        if label == OPEN_PALM and not still:
            label = PALM_MOVING
        return label, self.confirm.update(t, label)
