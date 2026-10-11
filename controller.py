"""
Traffic signal state machine for a two-road junction:
  NS = North-South road, EW = East-West road, PED = pedestrian WALK light.

Safety rules are enforced here, so no gesture can create a dangerous state:
  * a green light always goes through AMBER before turning red
  * an ALL-RED clearance follows before any new green or WALK
  * the two roads are never green together, and WALK never shows with a green
"""
from dataclasses import dataclass

R, Y, G, OFF = "R", "Y", "G", "OFF"

AUTO = "AUTO"
GO_NS = "GO_NS"
GO_EW = "GO_EW"
ALL_STOP = "ALL_STOP"
PEDESTRIAN = "PEDESTRIAN"
CAUTION = "CAUTION"

MODE_TEXT = {
    AUTO: "AUTO TIMER",
    GO_NS: "MANUAL: NORTH-SOUTH GO",
    GO_EW: "MANUAL: EAST-WEST GO",
    ALL_STOP: "ALL STOP",
    PEDESTRIAN: "PEDESTRIAN CROSSING",
    CAUTION: "FLASHING AMBER (CAUTION)",
}


@dataclass(frozen=True)
class Lamps:
    ns: str
    ew: str
    ped: bool

    def is_safe(self):
        if self.ns == G and self.ew == G:
            return False
        if self.ped and (self.ns == G or self.ew == G):
            return False
        return True

    def bits(self):
        """7 characters for the Arduino: NS R,Y,G  EW R,Y,G  PED."""
        out = ""
        for lamp in (self.ns, self.ew):
            out += "".join("1" if lamp == c else "0" for c in (R, Y, G))
        return out + ("1" if self.ped else "0")


ALL_RED = Lamps(R, R, False)


class SignalController:
    def __init__(self, now=0.0, green=8.0, yellow=2.0, clearance=1.0):
        self.green, self.yellow, self.clearance = green, yellow, clearance
        self.mode = AUTO
        self.mode_start = now
        self.steps = []           # pending transition: [(Lamps, end_time, text)]
        self.lamps = self._steady(AUTO, now)

    # ---- what each mode shows once it is running ----
    def _steady(self, mode, now):
        dt = max(0.0, now - self.mode_start)
        if mode == GO_NS:
            return Lamps(G, R, False)
        if mode == GO_EW:
            return Lamps(R, G, False)
        if mode == ALL_STOP:
            return ALL_RED
        if mode == PEDESTRIAN:
            return Lamps(R, R, True)
        if mode == CAUTION:
            on = int(dt * 2) % 2 == 0          # 1 Hz blink
            return Lamps(Y if on else OFF, Y if on else OFF, False)
        # AUTO: NS green -> NS amber -> all red -> EW green -> EW amber -> all red
        half = self.green + self.yellow + self.clearance
        p = dt % (2 * half)
        road_a = p < half
        q = p if road_a else p - half
        lamp = G if q < self.green else (Y if q < self.green + self.yellow else R)
        return Lamps(lamp, R, False) if road_a else Lamps(R, lamp, False)

    def transitioning(self):
        return bool(self.steps)

    def phase_text(self):
        return self.steps[0][2] if self.steps else ""

    def update(self, now):
        while self.steps and now >= self.steps[0][1]:
            self.steps.pop(0)
        self.lamps = self.steps[0][0] if self.steps else self._steady(self.mode, now)
        assert self.lamps.is_safe(), self.lamps
        return self.lamps

    def request(self, mode, now):
        """Switch to `mode` safely. Returns False if already in that mode."""
        cur = self.update(now)
        if mode == self.mode and not self.steps:
            return False

        # what the new mode will show first
        saved = self.mode_start
        self.mode_start = now
        target = self._steady(mode, now)
        self.mode_start = saved

        steps, t = [], now
        # 1. any lit green/amber that will not stay the same goes AMBER first
        ns_clear = cur.ns in (G, Y) and target.ns != cur.ns
        ew_clear = cur.ew in (G, Y) and target.ew != cur.ew
        if ns_clear or ew_clear:
            amber = Lamps(Y if ns_clear else (R if cur.ns == OFF else cur.ns),
                          Y if ew_clear else (R if cur.ew == OFF else cur.ew),
                          False)
            t += self.yellow
            steps.append((amber, t, "Clearing: amber"))
            cur = Lamps(R if ns_clear else amber.ns, R if ew_clear else amber.ew, False)

        # 2. ALL-RED clearance before a new green or WALK
        new_green = (target.ns == G and cur.ns != G) or (target.ew == G and cur.ew != G)
        new_walk = target.ped and not cur.ped
        if (new_green or new_walk) and (steps or cur != ALL_RED):
            t += self.clearance
            steps.append((ALL_RED, t, "All-red clearance"))
        elif cur.ped and not target.ped:
            t += self.clearance
            steps.append((ALL_RED, t, "Pedestrians clearing"))

        self.mode = mode
        self.mode_start = t
        self.steps = steps
        self.update(now)
        return True
