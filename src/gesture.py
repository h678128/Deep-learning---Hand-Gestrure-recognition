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
# MediaPipe: 0=wrist, 1=thumb_cmc, 4=thumb_tip, 5=index_mcp, 8=index_tip,
#            9=middle_mcp, 12=middle_tip, 13=ring_mcp, 16=ring_tip, 17=pinky_mcp, 20=pinky_tip
WRIST      = 0
THUMB_CMC  = 1
THUMB_TIP  = 2
INDEX_MCP  = 3
INDEX_TIP  = 4
MIDDLE_MCP = 5
MIDDLE_TIP = 6
RING_MCP   = 7
RING_TIP   = 8
PINKY_MCP  = 9
PINKY_TIP  = 10

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


def _palm_center(lm: np.ndarray) -> np.ndarray:
    pts = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    return lm[pts].mean(axis=0)


def _hand_scale(lm: np.ndarray) -> float:
    return max(1.0, _dist(lm[WRIST], lm[MIDDLE_MCP]))


def classify_gesture(landmarks: np.ndarray) -> str:
    pc    = _palm_center(landmarks)
    scale = _hand_scale(landmarks)

    i_d = _dist(landmarks[INDEX_TIP],  pc) / scale
    m_d = _dist(landmarks[MIDDLE_TIP], pc) / scale
    r_d = _dist(landmarks[RING_TIP],   pc) / scale
    p_d = _dist(landmarks[PINKY_TIP],  pc) / scale
    t_d = _dist(landmarks[THUMB_TIP],  pc) / scale

    # Åpen hånd: alle fingertuppper langt fra palmen
    if i_d > 0.8 and m_d > 0.8 and r_d > 0.8 and p_d > 0.8:
        return OPEN_HAND

    # Knyttneve: alle fingertuppper samlet nær palmen
    if i_d < 0.65 and m_d < 0.65 and r_d < 0.7 and p_d < 0.7:
        return FIST

    # Fredstegn/høyreklikk: peke+langefinger ute, ring+lillefinger+tommel inne
    if i_d > 0.8 and m_d > 0.8 and r_d < 0.6 and p_d < 0.6:
        return RIGHT_CLICK if t_d > 0.6 else "scroll"

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
