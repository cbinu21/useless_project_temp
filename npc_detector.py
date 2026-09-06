import cv2
import time
import math
import mediapipe as mp

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# NPC DETECTOR™
# Real Computer Vision + Ridiculous NPC Analysis
# =========================================================


# =========================================================
# SETTINGS
# =========================================================

MOVEMENT_DURATION = 6
BLINK_DURATION = 8
SMILE_CALIBRATION = 2
SMILE_DURATION = 5


# =========================================================
# MEDIAPIPE FACE LANDMARKER
# =========================================================

base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,
    running_mode=vision.RunningMode.VIDEO
)

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# =========================================================
# EYE LANDMARKS
# =========================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


def eye_aspect_ratio(landmarks, eye):

    p1 = landmarks[eye[0]]
    p2 = landmarks[eye[1]]
    p3 = landmarks[eye[2]]
    p4 = landmarks[eye[3]]
    p5 = landmarks[eye[4]]
    p6 = landmarks[eye[5]]

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)

    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 0

    return (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)


def get_npc_class(score):

    if score >= 85:
        return "BACKGROUND VILLAGER"

    elif score >= 70:
        return "SHOPKEEPER NPC"

    elif score >= 55:
        return "QUEST NPC"

    elif score >= 35:
        return "PLAYABLE CHARACTER"

    else:
        return "CHAOTIC PLAYER"


def clamp(value, minimum, maximum):

    return max(
        minimum,
        min(maximum, value)
    )


# =========================================================
# RETRO PIXEL UI HELPERS
# =========================================================

UI_GREEN = (80, 255, 170)
UI_CYAN = (255, 220, 90)
UI_WHITE = (245, 245, 245)
UI_DIM = (150, 165, 170)
UI_RED = (90, 100, 255)
UI_DARK = (10, 14, 16)


def pixel_text(frame, text, y, scale=1.0, thickness=2, color=UI_WHITE, x=None):
    """Crisp, slightly pixelated terminal-style centered text."""
    font = cv2.FONT_HERSHEY_PLAIN
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    if x is None:
        x = (frame.shape[1] - text_size[0]) // 2
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_8)


def panel(frame, x1, y1, x2, y2, fill=(12, 18, 20), border=UI_GREEN, thickness=2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, thickness)
    # Small corner cuts for a retro HUD feel.
    cut = 10
    cv2.line(frame, (x1, y1), (x1 + cut, y1), border, thickness)
    cv2.line(frame, (x2 - cut, y2), (x2, y2), border, thickness)


def progress_bar(frame, x, y, width, height, value, label=None):
    value = clamp(float(value), 0, 100)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (25, 32, 34), -1)
    filled = int(width * value / 100)
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + height), UI_GREEN, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), UI_DIM, 1)
    if label:
        pixel_text(frame, label, y - 7, 1.1, 1, UI_DIM, x=x)


def scanlines(frame, step=4, alpha=0.08):
    overlay = frame.copy()
    for y in range(0, frame.shape[0], step):
        cv2.line(overlay, (0, y), (frame.shape[1], y), (0, 0, 0), 1)
    return cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)


# =========================================================
# MAIN APPLICATION
# =========================================================

class NPCDetector:

    def __init__(self):

        # -------------------------------------------------
        # CAMERA
        # -------------------------------------------------

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open webcam."
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

        # -------------------------------------------------
        # APPLICATION STATE
        # -------------------------------------------------

        self.running = True

        self.state = "WELCOME"

        self.state_start = time.time()

        self.timestamp = 0

        # -------------------------------------------------
        # SCORES
        # -------------------------------------------------

        self.movement_score = 0
        self.blink_score = 0
        self.smile_score = 0

        self.final_score = 0

        # -------------------------------------------------
        # MOVEMENT DATA
        # -------------------------------------------------

        self.position_history = deque(
            maxlen=60
        )

        self.movement_values = []

        # -------------------------------------------------
        # BLINK DATA
        # -------------------------------------------------

        self.blink_count = 0

        self.eye_closed = False

        # -------------------------------------------------
        # SMILE DATA
        # -------------------------------------------------

        self.smile_baseline = None

        self.smile_values = []

        self.smile_triggered = False

        self.smile_start = None

        self.smile_reaction = None

    # =====================================================
    # STATE CHANGE
    # =====================================================

    def change_state(self, new_state):

        self.state = new_state

        self.state_start = time.time()

        # Reset movement

        if new_state == "MOVEMENT":

            self.position_history.clear()

            self.movement_values.clear()

            self.movement_score = 0

        # Reset blink

        elif new_state == "BLINK":

            self.blink_count = 0

            self.eye_closed = False

            self.blink_score = 0

        # Reset smile

        elif new_state == "SMILE":

            self.smile_baseline = None

            self.smile_values.clear()

            self.smile_triggered = False

            self.smile_start = None

            self.smile_reaction = None

            self.smile_score = 0

        # Reset final

        elif new_state == "WELCOME":

            self.final_score = 0

    # =====================================================
    # KEYBOARD
    # =====================================================

    def handle_key(self, key):

        # Q = quit

        if key == ord("q"):

            self.running = False

            return

        # R = restart

        if key == ord("r"):

            self.change_state(
                "WELCOME"
            )

            return

        # SPACE = start ONLY

        if key == 32:

            if self.state == "WELCOME":

                self.change_state(
                    "MOVEMENT"
                )

    # =====================================================
    # TEXT DRAWING
    # =====================================================

    def draw_text(self, frame, message, y, scale=1.0, thickness=2, color=UI_WHITE, x=None):
        pixel_text(frame, message, y, scale, thickness, color, x)

    def header(self, frame, section, index=None):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 58), (7, 12, 14), -1)
        cv2.line(frame, (0, 57), (w, 57), UI_GREEN, 2)
        pixel_text(frame, "NPC DETECTOR", 36, 1.7, 2, UI_GREEN, x=32)
        right = section if index is None else f"TEST {index:02d} // {section}"
        text_size, _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_PLAIN, 1.25, 1)
        pixel_text(frame, right, 34, 1.25, 1, UI_DIM, x=w - text_size[0] - 32)

    def footer(self, frame, message="R RESTART    Q QUIT"):
        h, w = frame.shape[:2]
        cv2.line(frame, (28, h - 45), (w - 28, h - 45), (50, 65, 68), 1)
        pixel_text(frame, message, h - 18, 1.0, 1, UI_DIM)

    # =====================================================
    # WELCOME SCREEN
    # =====================================================

    def draw_welcome(self, frame):
        h, w = frame.shape[:2]
        self.header(frame, "SYSTEM READY")

        # Center console.
        panel(frame, 120, 120, w - 120, h - 120, (9, 15, 17), UI_GREEN, 2)
        pixel_text(frame, "// HUMANITY VERIFICATION SYSTEM //", 175, 1.5, 2, UI_DIM)
        pixel_text(frame, "NPC", 280, 5.0, 6, UI_GREEN)
        pixel_text(frame, "DETECTOR", 345, 3.0, 4, UI_WHITE)
        pixel_text(frame, "ARE YOU ACTUALLY HUMAN?", 405, 1.6, 2, UI_CYAN)

        # Fake diagnostic readout.
        pixel_text(frame, "FACE      [ READY ]", 485, 1.15, 1, UI_DIM, x=190)
        pixel_text(frame, "FREE WILL [ UNKNOWN ]", 520, 1.15, 1, UI_DIM, x=190)
        pixel_text(frame, "NPC INDEX [ ARMED  ]", 555, 1.15, 1, UI_DIM, x=190)

        cv2.rectangle(frame, (w - 410, 475), (w - 180, 530), (16, 30, 26), -1)
        cv2.rectangle(frame, (w - 410, 475), (w - 180, 530), UI_GREEN, 2)
        pixel_text(frame, "[ SPACE ]", 512, 1.7, 2, UI_GREEN, x=w - 375)
        self.footer(frame, "SPACE START    R RESTART    Q QUIT")

    # =====================================================
    # MOVEMENT TEST
    # =====================================================

    def movement_test(self, frame, landmarks):
        elapsed = time.time() - self.state_start
        remaining = max(0, MOVEMENT_DURATION - int(elapsed))
        h, w = frame.shape[:2]
        self.header(frame, "MOVEMENT ANALYSIS", 1)

        pixel_text(frame, "MOVE YOUR HEAD NATURALLY", 105, 1.55, 2, UI_WHITE)
        progress_bar(frame, 300, 125, w - 600, 14, elapsed / MOVEMENT_DURATION * 100)
        pixel_text(frame, f"00:{remaining:02d}", 180, 2.0, 2, UI_CYAN)

        if landmarks:
            nose = landmarks[1]
            x, y = nose.x, nose.y
            self.position_history.append((x, y))
            if len(self.position_history) >= 2:
                previous_x, previous_y = self.position_history[-2]
                movement = math.sqrt((x - previous_x) ** 2 + (y - previous_y) ** 2)
                self.movement_values.append(movement)

            center_x = int(nose.x * w)
            center_y = int(nose.y * h)
            cv2.circle(frame, (center_x, center_y), 12, UI_GREEN, 2)
            cv2.circle(frame, (center_x, center_y), 3, UI_GREEN, -1)
            pixel_text(frame, "● FACE TRACKED", h - 85, 1.15, 1, UI_GREEN, x=35)
        else:
            pixel_text(frame, "! NO FACE DETECTED", h - 85, 1.15, 1, UI_RED, x=35)

        current = 0
        if self.movement_values:
            current = clamp(100 - (sum(self.movement_values[-30:]) / len(self.movement_values[-30:])) * 300, 0, 100)
        panel(frame, 35, h - 185, 360, h - 105, (9, 15, 17), (50, 65, 68), 1)
        pixel_text(frame, "STILLNESS INDEX", h - 150, 1.0, 1, UI_DIM, x=55)
        progress_bar(frame, 55, h - 135, 280, 12, current)
        pixel_text(frame, f"{current:05.1f}%", h - 110, 1.0, 1, UI_WHITE, x=55)
        self.footer(frame)

        if elapsed >= MOVEMENT_DURATION:
            if self.movement_values:
                average_movement = sum(self.movement_values) / len(self.movement_values)
                self.movement_score = clamp(100 - average_movement * 300, 0, 100)
            else:
                self.movement_score = 100
            self.change_state("BLINK")

    # =====================================================
    # BLINK TEST
    # =====================================================

    def blink_test(self, frame, landmarks):
        elapsed = time.time() - self.state_start
        remaining = max(0, BLINK_DURATION - int(elapsed))
        h, w = frame.shape[:2]
        self.header(frame, "BLINK ANALYSIS", 2)
        pixel_text(frame, "BLINK NATURALLY", 105, 1.55, 2, UI_WHITE)
        progress_bar(frame, 300, 125, w - 600, 14, elapsed / BLINK_DURATION * 100)
        pixel_text(frame, f"00:{remaining:02d}", 180, 2.0, 2, UI_CYAN)

        ear = 0.0
        if landmarks:
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2
            if ear < 0.21:
                if not self.eye_closed:
                    self.blink_count += 1
                self.eye_closed = True
            else:
                self.eye_closed = False

            status = "EYES CLOSED" if ear < 0.21 else "EYES OPEN"
            status_color = UI_CYAN if ear < 0.21 else UI_GREEN
            pixel_text(frame, status, 300, 2.0, 2, status_color)
            pixel_text(frame, f"EYE RATIO  {ear:.3f}", 350, 1.15, 1, UI_DIM)
        else:
            pixel_text(frame, "! NO FACE DETECTED", 300, 1.4, 2, UI_RED)

        panel(frame, 35, h - 185, 420, h - 105, (9, 15, 17), (50, 65, 68), 1)
        pixel_text(frame, "BLINK COUNTER", h - 150, 1.0, 1, UI_DIM, x=55)
        pixel_text(frame, f"{self.blink_count:02d}", h - 112, 2.2, 2, UI_GREEN, x=55)
        pixel_text(frame, "/ 8 SEC", h - 112, 1.0, 1, UI_DIM, x=105)
        self.footer(frame)

        if elapsed >= BLINK_DURATION:
            blink_rate = self.blink_count * 60 / BLINK_DURATION
            if blink_rate < 3:
                self.blink_score = 90
            elif blink_rate < 7:
                self.blink_score = 65
            elif blink_rate <= 25:
                self.blink_score = 25
            else:
                self.blink_score = 60
            self.change_state("SMILE")

    # =====================================================
    # SMILE TEST
    # =====================================================

    def smile_test(self, frame, landmarks):
        elapsed = time.time() - self.state_start
        h, w = frame.shape[:2]
        self.header(frame, "REACTION ANALYSIS", 3)

        if elapsed < SMILE_CALIBRATION:
            pixel_text(frame, "CALIBRATING EXPRESSION BASELINE", 170, 1.45, 2, UI_WHITE)
            pixel_text(frame, "DO NOT SMILE YET", 235, 1.5, 2, UI_CYAN)
            progress_bar(frame, 300, 275, w - 600, 14, elapsed / SMILE_CALIBRATION * 100)
            if landmarks:
                mouth_width = distance(landmarks[61], landmarks[291])
                face_width = distance(landmarks[234], landmarks[454])
                if face_width > 0:
                    self.smile_values.append(mouth_width / face_width)
            self.footer(frame)
            return

        if self.smile_baseline is None:
            if self.smile_values:
                self.smile_baseline = sum(self.smile_values) / len(self.smile_values)
            else:
                self.smile_baseline = 0.35
            self.smile_start = time.time()

        test_elapsed = time.time() - self.smile_start
        pixel_text(frame, "SMILE!", 190, 3.5, 4, UI_GREEN if not self.smile_triggered else UI_CYAN)
        pixel_text(frame, "HOW FAST CAN YOU LOOK HUMAN?", 245, 1.35, 2, UI_WHITE)
        progress_bar(frame, 300, 275, w - 600, 14, test_elapsed / SMILE_DURATION * 100)

        if landmarks:
            mouth_width = distance(landmarks[61], landmarks[291])
            face_width = distance(landmarks[234], landmarks[454])
            if face_width > 0:
                ratio = mouth_width / face_width
                threshold = self.smile_baseline * 1.18
                if ratio > threshold and not self.smile_triggered:
                    self.smile_triggered = True
                    self.smile_reaction = test_elapsed
                    reaction = self.smile_reaction
                    if reaction < 0.4:
                        self.smile_score = 90
                    elif reaction < 0.8:
                        self.smile_score = 65
                    elif reaction < 1.5:
                        self.smile_score = 35
                    elif reaction < 3:
                        self.smile_score = 70
                    else:
                        self.smile_score = 85

        if self.smile_triggered:
            pixel_text(frame, "✓ EXPRESSION ACCEPTED", 365, 1.4, 2, UI_GREEN)
            pixel_text(frame, f"REACTION TIME  {self.smile_reaction:.2f}s", 410, 1.2, 1, UI_DIM)
        else:
            pixel_text(frame, "WAITING FOR HUMAN RESPONSE...", 365, 1.1, 1, UI_DIM)

        self.footer(frame)
        if test_elapsed >= SMILE_DURATION:
            if not self.smile_triggered:
                self.smile_score = 90
            self.change_state("ANALYZING")

    # =====================================================
    # ANALYSIS
    # =====================================================

    def analyzing(self, frame):
        elapsed = time.time() - self.state_start
        h, w = frame.shape[:2]
        self.header(frame, "NEURAL JUDGEMENT")
        panel(frame, 170, 130, w - 170, h - 120, (7, 12, 14), UI_GREEN, 2)
        pixel_text(frame, "ANALYZING HUMANITY", 215, 2.3, 3, UI_WHITE)
        steps = [
            "MEASURING UNNECESSARY MOVEMENT",
            "COUNTING SUSPICIOUS BLINKS",
            "CALCULATING FREE-WILL RESPONSE",
            "CONSULTING NPC DATABASE",
            "GENERATING ABSOLUTELY UNRELIABLE VERDICT",
        ]
        active = min(len(steps) - 1, int(elapsed * 1.5))
        for i, step in enumerate(steps):
            y = 285 + i * 48
            if i < active:
                mark, color = "[OK]", UI_GREEN
            elif i == active:
                mark, color = "[>>]", UI_CYAN
            else:
                mark, color = "[  ]", UI_DIM
            pixel_text(frame, f"{mark} {step}", y, 1.05, 1, color, x=235)
        progress_bar(frame, 235, h - 180, w - 470, 16, min(100, elapsed / 3 * 100))
        pixel_text(frame, "PLEASE REMAIN SUSPICIOUS", h - 140, 1.1, 1, UI_DIM)

        if elapsed >= 3:
            self.final_score = (self.movement_score * 0.40 + self.blink_score * 0.30 + self.smile_score * 0.30)
            self.change_state("RESULT")

    # =====================================================
    # RESULT SCREEN
    # =====================================================

    def result(self, frame):
        npc_class = get_npc_class(self.final_score)
        h, w = frame.shape[:2]
        self.header(frame, "FINAL VERDICT")

        panel(frame, 55, 90, w - 55, h - 75, (7, 12, 14), UI_GREEN, 2)
        pixel_text(frame, "HUMANITY REPORT", 140, 2.0, 3, UI_WHITE)
        pixel_text(frame, f"{self.final_score:.1f}%", 250, 5.0, 6, UI_GREEN)
        pixel_text(frame, "NPC PROBABILITY", 292, 1.25, 2, UI_CYAN)

        # Class badge.
        badge_w = min(700, w - 160)
        bx = (w - badge_w) // 2
        cv2.rectangle(frame, (bx, 330), (bx + badge_w, 395), (15, 28, 25), -1)
        cv2.rectangle(frame, (bx, 330), (bx + badge_w, 395), UI_GREEN, 2)
        pixel_text(frame, npc_class, 373, 2.0, 2, UI_GREEN)

        labels = [
            ("MOVEMENT", self.movement_score),
            ("BLINK", self.blink_score),
            ("REACTION", self.smile_score),
        ]
        y = 455
        for label, value in labels:
            pixel_text(frame, f"{label:<10} {value:05.1f}", y, 1.0, 1, UI_DIM, x=150)
            progress_bar(frame, 430, y - 12, min(480, w - 650), 12, value)
            y += 42

        # Ridiculous verdict text.
        if npc_class == "BACKGROUND VILLAGER":
            verdict = "LIKELY TO REPEAT THE SAME DIALOGUE FOREVER."
        elif npc_class == "SHOPKEEPER NPC":
            verdict = "HAS 37 POTIONS AND NO PERSONALITY UPGRADES."
        elif npc_class == "QUEST NPC":
            verdict = "KNOWS SOMETHING IMPORTANT. WILL NOT EXPLAIN IT."
        elif npc_class == "PLAYABLE CHARACTER":
            verdict = "FREE WILL DETECTED. DEVELOPER SLIGHTLY CONCERNED."
        else:
            verdict = "TOO MUCH FREE WILL. NPC SYSTEM HAS LOST CONTROL."
        pixel_text(frame, verdict, h - 125, 1.0, 1, UI_WHITE)
        self.footer(frame, "R TEST AGAIN    Q QUIT")

    # =====================================================
    # MAIN LOOP
    # =====================================================

    def run(self):
        cv2.namedWindow("NPC DETECTOR", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("NPC DETECTOR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while self.running:

            success, frame = self.cap.read()

            if not success:
                continue

            # Mirror camera

            frame = cv2.flip(
                frame,
                1
            )

            # -------------------------------------------------
            # MEDIAPIPE IMAGE
            # -------------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # IMPORTANT:
            # Image belongs to mediapipe,
            # NOT mediapipe.tasks.python.vision

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            self.timestamp += 1

            result = (
                landmarker.detect_for_video(
                    mp_image,
                    self.timestamp
                )
            )

            landmarks = None

            if result.face_landmarks:

                landmarks = (
                    result.face_landmarks[0]
                )

            # -------------------------------------------------
            # SUBTLE RETRO DISPLAY EFFECT
            # -------------------------------------------------
            frame = scanlines(frame, step=4, alpha=0.10)

            # -------------------------------------------------
            # STATE MACHINE
            # -------------------------------------------------

            if self.state == "WELCOME":

                self.draw_welcome(
                    frame
                )

            elif self.state == "MOVEMENT":

                self.movement_test(
                    frame,
                    landmarks
                )

            elif self.state == "BLINK":

                self.blink_test(
                    frame,
                    landmarks
                )

            elif self.state == "SMILE":

                self.smile_test(
                    frame,
                    landmarks
                )

            elif self.state == "ANALYZING":

                self.analyzing(
                    frame
                )

            elif self.state == "RESULT":

                self.result(
                    frame
                )

            # -------------------------------------------------
            # SHOW
            # -------------------------------------------------

            cv2.imshow(
                "NPC DETECTOR",
                frame
            )

            # -------------------------------------------------
            # KEYBOARD
            # -------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key != 255:

                self.handle_key(
                    key
                )

        self.cap.release()

        cv2.destroyAllWindows()


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    app = NPCDetector()

    app.run()