from __future__ import annotations

import time

import numpy as np

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

# Indekser i vår 11-punkts modell (LANDMARK_INDICES_11 = 0,1,4,5,8,9,12,13,16,17,20)
WRIST      = 0
INDEX_MCP  = 1
INDEX_TIP  = 2
MIDDLE_MCP = 3
MIDDLE_TIP = 4
RING_MCP   = 5
RING_TIP   = 6
PINKY_MCP  = 7
PINKY_TIP  = 8
THUMB_IP   = 9
THUMB_TIP  = 10

OPEN_HAND   = "open_hand"
FIST        = "fist"
RIGHT_CLICK = "right_click"
UNKNOWN     = "unknown"

GESTURE_LABELS = {
    OPEN_HAND:   "Musemodus",
    FIST:        "Venstreklikk",
    RIGHT_CLICK: "Hoyreklikk",
    UNKNOWN:     "",
}


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _finger_extended(lm: np.ndarray, tip: int, mcp: int) -> bool:
    return _dist(lm[tip], lm[WRIST]) > _dist(lm[mcp], lm[WRIST]) * 1.7


def _finger_curled(lm: np.ndarray, tip: int, mcp: int) -> bool:
    return _dist(lm[tip], lm[WRIST]) < _dist(lm[mcp], lm[WRIST]) * 1.3


def _thumb_out(lm: np.ndarray) -> bool:
    hand_height = max(1.0, abs(float(lm[INDEX_MCP, 1]) - float(lm[WRIST, 1])))
    x_dist = abs(float(lm[THUMB_TIP, 0]) - float(lm[WRIST, 0]))
    return x_dist > hand_height * 0.5


def classify_gesture(landmarks: np.ndarray) -> str:
    i_up  = _finger_extended(landmarks, INDEX_TIP,  INDEX_MCP)
    m_up  = _finger_extended(landmarks, MIDDLE_TIP, MIDDLE_MCP)
    r_up  = _finger_extended(landmarks, RING_TIP,   RING_MCP)
    p_up  = _finger_extended(landmarks, PINKY_TIP,  PINKY_MCP)
    r_dn  = _finger_curled(landmarks,   RING_TIP,   RING_MCP)
    p_dn  = _finger_curled(landmarks,   PINKY_TIP,  PINKY_MCP)
    thumb = _thumb_out(landmarks)

    if i_up and m_up and r_up and p_up:
        return OPEN_HAND
    if not i_up and not m_up and r_dn and p_dn:
        return FIST
    if thumb and i_up and m_up and r_dn and p_dn:
        return RIGHT_CLICK
    if not thumb and i_up and m_up and r_dn and p_dn:
        return "scroll"
    return UNKNOWN


class GestureController:
    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        click_cooldown: float = 0.8,
        mouse_smoothing: float = 0.5,
    ) -> None:
        if PYAUTOGUI_AVAILABLE:
            self.screen_w, self.screen_h = pyautogui.size()
        else:
            self.screen_w, self.screen_h = 1920, 1080

        self.frame_w = frame_w
        self.frame_h = frame_h
        self.click_cooldown = click_cooldown
        self.mouse_smoothing = mouse_smoothing

        self._prev_gesture = UNKNOWN
        self._last_click_time = 0.0
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None

    def process(self, landmarks: np.ndarray, gesture: str) -> None:
        if not PYAUTOGUI_AVAILABLE:
            return

        now = time.time()

        if gesture == OPEN_HAND:
            tip = landmarks[INDEX_TIP]
            raw_x = float(tip[0]) / self.frame_w * self.screen_w
            raw_y = float(tip[1]) / self.frame_h * self.screen_h

            if self._smooth_x is None:
                self._smooth_x, self._smooth_y = raw_x, raw_y
            else:
                a = self.mouse_smoothing
                self._smooth_x = a * self._smooth_x + (1 - a) * raw_x
                self._smooth_y = a * self._smooth_y + (1 - a) * raw_y

            pyautogui.moveTo(
                int(np.clip(self._smooth_x, 0, self.screen_w - 1)),
                int(np.clip(self._smooth_y, 0, self.screen_h - 1)),
                _pause=False,
            )

        elif gesture == FIST and self._prev_gesture != FIST:
            if now - self._last_click_time > self.click_cooldown:
                pyautogui.click()
                self._last_click_time = now

        elif gesture == RIGHT_CLICK and self._prev_gesture != RIGHT_CLICK:
            if now - self._last_click_time > self.click_cooldown:
                pyautogui.rightClick()
                self._last_click_time = now

        self._prev_gesture = gesture

    def reset(self) -> None:
        self._smooth_x = None
        self._smooth_y = None
        self._prev_gesture = UNKNOWN
