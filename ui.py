"""On-screen dashboard: camera with hand skeleton, live junction view, status and log."""
import cv2
import numpy as np

from controller import R, Y, G, OFF, MODE_TEXT
from gestures import (GESTURE_NAMES, GESTURE_TO_COMMAND, STATIC_COMMAND_GESTURES,
                      OPEN_PALM, POINT_UP, POINT_SIDE, V_SIGN, FIST, WAVE)

CAM_W, CAM_H = 640, 480
PANEL_W = 420
LEGEND_H = 130
W, H = CAM_W + PANEL_W, CAM_H + LEGEND_H

FONT = cv2.FONT_HERSHEY_SIMPLEX
BOLD = cv2.FONT_HERSHEY_DUPLEX

BG = (28, 24, 22)
CARD = (45, 40, 37)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)
RED = (60, 60, 230)
AMBER = (0, 190, 255)
GREEN = (90, 210, 70)
DIM = (60, 60, 60)
ACCENT = (255, 190, 60)

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

LEGEND = [
    (OPEN_PALM, "Open palm (still)"),
    (POINT_UP, "Point up"),
    (POINT_SIDE, "Point sideways"),
    (V_SIGN, "V sign"),
    (FIST, "Fist"),
    (WAVE, "Wave side to side"),
]

COMMAND_TEXT = {
    "ALL_STOP": "ALL STOP",
    "GO_NS": "North-South GO",
    "GO_EW": "East-West GO",
    "PEDESTRIAN": "Pedestrians WALK",
    "AUTO": "Back to AUTO timer",
    "CAUTION": "Flashing amber",
}


def _lamp_color(state, kind):
    on = {R: RED, Y: AMBER, G: GREEN}[kind]
    return on if state == kind else DIM


def _text(img, s, xy, scale=0.55, color=TEXT, thick=1, font=FONT):
    cv2.putText(img, s, xy, font, scale, color, thick, cv2.LINE_AA)


def draw_hand(img, lm, color):
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(map(int, lm[a])), tuple(map(int, lm[b])), color, 2, cv2.LINE_AA)
    for p in lm:
        cv2.circle(img, tuple(map(int, p)), 4, (255, 255, 255), -1, cv2.LINE_AA)


def _signal_head(img, x, y, state, vertical=True):
    """Three-lamp signal head with top-left at (x, y)."""
    r, gap = 13, 32
    w, h = (36, 100) if vertical else (100, 36)
    cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)
    for i, kind in enumerate((R, Y, G)):
        cx, cy = (x + 18, y + 18 + i * gap) if vertical else (x + 18 + i * gap, y + 18)
        cv2.circle(img, (cx, cy), r, _lamp_color(state, kind), -1, cv2.LINE_AA)


def draw_junction(img, x0, y0, lamps):
    """Top-down junction with both signal heads and the pedestrian light."""
    w, h = PANEL_W - 40, 200
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (40, 70, 45), -1)       # grass
    cx, cy = x0 + w // 2, y0 + h // 2
    road = 54
    cv2.rectangle(img, (cx - road // 2, y0), (cx + road // 2, y0 + h), (70, 70, 70), -1)   # NS road
    cv2.rectangle(img, (x0, cy - road // 2), (x0 + w, cy + road // 2), (70, 70, 70), -1)   # EW road
    for yy in range(y0 + 4, y0 + h, 18):                                     # lane markings
        if abs(yy - cy) > road // 2:
            cv2.line(img, (cx, yy), (cx, yy + 8), (200, 200, 200), 1)
    for xx in range(x0 + 4, x0 + w, 18):
        if abs(xx - cx) > road // 2:
            cv2.line(img, (xx, cy), (xx + 8, cy), (200, 200, 200), 1)
    # zebra crossing (pedestrian)
    zc = (255, 255, 255) if lamps.ped else (120, 120, 120)
    for i in range(5):
        xx = cx - road // 2 + 4 + i * 10
        cv2.rectangle(img, (xx, cy - road // 2 - 16), (xx + 5, cy - road // 2 - 2), zc, -1)

    _signal_head(img, x0 + 12, y0 + 10, lamps.ns, vertical=True)
    _text(img, "N-S", (x0 + 14, y0 + 128), 0.45, TEXT)
    _signal_head(img, x0 + w - 112, y0 + h - 48, lamps.ew, vertical=False)
    _text(img, "E-W", (x0 + w - 150, y0 + h - 24), 0.45, TEXT)

    # pedestrian light
    px, py = x0 + w - 112, y0 + 10
    color = GREEN if lamps.ped else RED
    cv2.rectangle(img, (px, py), (px + 100, py + 34), (20, 20, 20), -1)
    _text(img, "WALK" if lamps.ped else "DON'T WALK", (px + (24 if lamps.ped else 6), py + 23), 0.5, color, 1, BOLD)


def render(frame, *, lamps, mode, phase, label, progress, hand_lm, hand_state,
           fps, latency_ms, log, hardware, banner=None, flash=None):
    canvas = np.full((H, W, 3), BG, np.uint8)

    # ---------- camera ----------
    cam = cv2.resize(frame, (CAM_W, CAM_H))
    if hand_lm is not None:
        color = {"fire": GREEN, "hold": AMBER}.get(hand_state, (160, 160, 160))
        draw_hand(cam, hand_lm, color)
    canvas[:CAM_H, :CAM_W] = cam

    # detected gesture strip
    cv2.rectangle(canvas, (0, CAM_H - 46), (CAM_W, CAM_H), (0, 0, 0), -1)
    name = GESTURE_NAMES.get(label, label)
    lbl_color = GREEN if label in STATIC_COMMAND_GESTURES else MUTED
    _text(canvas, f"Detected: {name}", (12, CAM_H - 17), 0.65, lbl_color, 1, BOLD)
    if progress > 0:
        bx = 340
        cv2.rectangle(canvas, (bx, CAM_H - 32), (bx + 200, CAM_H - 16), (80, 80, 80), 1)
        cv2.rectangle(canvas, (bx, CAM_H - 32), (bx + int(200 * progress), CAM_H - 16), AMBER, -1)
        _text(canvas, "hold", (bx + 210, CAM_H - 18), 0.5, AMBER)

    _text(canvas, f"{fps:4.1f} FPS   {latency_ms:4.1f} ms/frame", (CAM_W - 250, 24), 0.55, (255, 255, 255))

    if flash:
        cv2.rectangle(canvas, (0, 0), (CAM_W - 1, CAM_H - 1), GREEN, 6)
        _text(canvas, flash, (14, 34), 0.9, GREEN, 2, BOLD)

    if banner:
        cv2.rectangle(canvas, (0, 60), (CAM_W, 150), (0, 0, 0), -1)
        _text(canvas, banner[0], (16, 98), 0.9, ACCENT, 2, BOLD)
        if len(banner) > 1:
            _text(canvas, banner[1], (16, 134), 0.6, TEXT)

    # ---------- legend under camera ----------
    y = CAM_H + 26
    _text(canvas, "GESTURES  (hold 1 s)", (12, y), 0.5, ACCENT, 1, BOLD)
    for i, (g, desc) in enumerate(LEGEND):
        col, row = i % 2, i // 2
        x, yy = 12 + col * 320, y + 28 + row * 30
        _text(canvas, f"{desc}", (x, yy), 0.5, TEXT)
        _text(canvas, f"-> {COMMAND_TEXT[GESTURE_TO_COMMAND[g]]}", (x + 150, yy), 0.5, MUTED)

    # ---------- right panel ----------
    px = CAM_W + 20
    _text(canvas, "SignalSync", (px, 38), 1.0, TEXT, 2, BOLD)
    _text(canvas, "Gesture-synced traffic signals", (px, 62), 0.5, MUTED)

    draw_junction(canvas, px, 80, lamps)

    cv2.rectangle(canvas, (px, 296), (px + PANEL_W - 40, 366), CARD, -1)
    _text(canvas, "MODE", (px + 12, 318), 0.45, MUTED)
    _text(canvas, MODE_TEXT[mode], (px + 12, 346), 0.6, TEXT, 1, BOLD)
    if phase:
        _text(canvas, phase, (px + 12, 362), 0.42, AMBER)

    hw_text = "Arduino: connected" if hardware else "Arduino: not connected (on-screen only)"
    _text(canvas, hw_text, (px, 390), 0.45, GREEN if hardware else MUTED)

    _text(canvas, "COMMAND LOG", (px, 420), 0.45, ACCENT, 1, BOLD)
    for i, line in enumerate(log[-6:][::-1]):
        _text(canvas, line, (px, 446 + i * 24), 0.48, TEXT if i == 0 else MUTED)

    _text(canvas, "Q quit   S screenshot", (px, H - 14), 0.45, MUTED)
    return canvas
