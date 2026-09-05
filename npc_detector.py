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

    def draw_text(
        self,
        frame,
        message,
        y,
        scale=1.0,
        thickness=2
    ):

        font = cv2.FONT_HERSHEY_SIMPLEX

        text_size, _ = cv2.getTextSize(
            message,
            font,
            scale,
            thickness
        )

        x = (
            frame.shape[1]
            - text_size[0]
        ) // 2

        cv2.putText(
            frame,
            message,
            (x, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    # =====================================================
    # WELCOME SCREEN
    # =====================================================

    def draw_welcome(self, frame):

        self.draw_text(
            frame,
            "NPC DETECTOR",
            180,
            2.0,
            4
        )

        self.draw_text(
            frame,
            "ARE YOU ACTUALLY HUMAN?",
            260,
            0.9,
            2
        )

        self.draw_text(
            frame,
            "PRESS SPACE TO BEGIN",
            390,
            0.8,
            2
        )

        self.draw_text(
            frame,
            "R = RESTART     Q = QUIT",
            470,
            0.55,
            1
        )

    # =====================================================
    # MOVEMENT TEST
    # =====================================================

    def movement_test(
        self,
        frame,
        landmarks
    ):

        elapsed = (
            time.time()
            - self.state_start
        )

        remaining = max(
            0,
            MOVEMENT_DURATION
            - int(elapsed)
        )

        self.draw_text(
            frame,
            "HUMANITY TEST 01",
            70,
            0.7,
            2
        )

        self.draw_text(
            frame,
            "MOVEMENT ANALYSIS",
            130,
            1.1,
            3
        )

        self.draw_text(
            frame,
            f"MOVE YOUR HEAD NATURALLY   [{remaining}]",
            220,
            0.65,
            2
        )

        # -------------------------------------------------
        # REAL MOVEMENT DETECTION
        # -------------------------------------------------

        if landmarks:

            nose = landmarks[1]

            x = nose.x
            y = nose.y

            self.position_history.append(
                (x, y)
            )

            if len(
                self.position_history
            ) >= 2:

                previous_x, previous_y = (
                    self.position_history[-2]
                )

                movement = math.sqrt(
                    (x - previous_x) ** 2
                    +
                    (y - previous_y) ** 2
                )

                self.movement_values.append(
                    movement
                )

            # Draw tracking point

            h, w = frame.shape[:2]

            center_x = int(
                nose.x * w
            )

            center_y = int(
                nose.y * h
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                8,
                (0, 255, 0),
                -1
            )

            self.draw_text(
                frame,
                "FACE TRACKED",
                300,
                0.6,
                2
            )

        else:

            self.draw_text(
                frame,
                "NO FACE DETECTED",
                300,
                0.7,
                2
            )

        # -------------------------------------------------
        # FINISH
        # -------------------------------------------------

        if elapsed >= MOVEMENT_DURATION:

            if self.movement_values:

                average_movement = (
                    sum(
                        self.movement_values
                    )
                    /
                    len(
                        self.movement_values
                    )
                )

                self.movement_score = clamp(
                    100
                    -
                    average_movement * 300,
                    0,
                    100
                )

            else:

                self.movement_score = 100

            self.change_state(
                "BLINK"
            )

    # =====================================================
    # BLINK TEST
    # =====================================================

    def blink_test(
        self,
        frame,
        landmarks
    ):

        elapsed = (
            time.time()
            - self.state_start
        )

        remaining = max(
            0,
            BLINK_DURATION
            - int(elapsed)
        )

        self.draw_text(
            frame,
            "HUMANITY TEST 02",
            70,
            0.7,
            2
        )

        self.draw_text(
            frame,
            "BLINK ANALYSIS",
            130,
            1.1,
            3
        )

        self.draw_text(
            frame,
            f"BLINK NATURALLY   [{remaining}]",
            220,
            0.7,
            2
        )

        # -------------------------------------------------
        # REAL BLINK DETECTION
        # -------------------------------------------------

        if landmarks:

            left_ear = eye_aspect_ratio(
                landmarks,
                LEFT_EYE
            )

            right_ear = eye_aspect_ratio(
                landmarks,
                RIGHT_EYE
            )

            ear = (
                left_ear
                +
                right_ear
            ) / 2

            if ear < 0.21:

                if not self.eye_closed:

                    self.blink_count += 1

                self.eye_closed = True

            else:

                self.eye_closed = False

            self.draw_text(
                frame,
                f"BLINKS DETECTED: {self.blink_count}",
                310,
                0.75,
                2
            )

        else:

            self.draw_text(
                frame,
                "NO FACE DETECTED",
                310,
                0.7,
                2
            )

        # -------------------------------------------------
        # FINISH
        # -------------------------------------------------

        if elapsed >= BLINK_DURATION:

            blink_rate = (
                self.blink_count
                * 60
                /
                BLINK_DURATION
            )

            if blink_rate < 3:

                self.blink_score = 90

            elif blink_rate < 7:

                self.blink_score = 65

            elif blink_rate <= 25:

                self.blink_score = 25

            else:

                self.blink_score = 60

            self.change_state(
                "SMILE"
            )

    # =====================================================
    # SMILE TEST
    # =====================================================

    def smile_test(
        self,
        frame,
        landmarks
    ):

        elapsed = (
            time.time()
            - self.state_start
        )

        # -------------------------------------------------
        # CALIBRATION
        # -------------------------------------------------

        if elapsed < SMILE_CALIBRATION:

            self.draw_text(
                frame,
                "HUMANITY TEST 03",
                70,
                0.7,
                2
            )

            self.draw_text(
                frame,
                "GET READY...",
                160,
                1.2,
                3
            )

            self.draw_text(
                frame,
                "DO NOT SMILE YET",
                250,
                0.75,
                2
            )

            if landmarks:

                mouth_width = distance(
                    landmarks[61],
                    landmarks[291]
                )

                face_width = distance(
                    landmarks[234],
                    landmarks[454]
                )

                if face_width > 0:

                    ratio = (
                        mouth_width
                        /
                        face_width
                    )

                    self.smile_values.append(
                        ratio
                    )

            return

        # -------------------------------------------------
        # BASELINE
        # -------------------------------------------------

        if self.smile_baseline is None:

            if self.smile_values:

                self.smile_baseline = (
                    sum(
                        self.smile_values
                    )
                    /
                    len(
                        self.smile_values
                    )
                )

            else:

                self.smile_baseline = 0.35

            self.smile_start = time.time()

        # -------------------------------------------------
        # SMILE TIMER
        # -------------------------------------------------

        test_elapsed = (
            time.time()
            -
            self.smile_start
        )

        self.draw_text(
            frame,
            "SMILE!",
            170,
            1.5,
            4
        )

        self.draw_text(
            frame,
            "HOW FAST CAN YOU LOOK HUMAN?",
            260,
            0.65,
            2
        )

        # -------------------------------------------------
        # REAL SMILE DETECTION
        # -------------------------------------------------

        if landmarks:

            mouth_width = distance(
                landmarks[61],
                landmarks[291]
            )

            face_width = distance(
                landmarks[234],
                landmarks[454]
            )

            if face_width > 0:

                ratio = (
                    mouth_width
                    /
                    face_width
                )

                threshold = (
                    self.smile_baseline
                    *
                    1.18
                )

                if (
                    ratio > threshold
                    and
                    not self.smile_triggered
                ):

                    self.smile_triggered = True

                    self.smile_reaction = (
                        test_elapsed
                    )

                    reaction = (
                        self.smile_reaction
                    )

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

                    self.draw_text(
                        frame,
                        f"REACTION: {reaction:.2f}s",
                        350,
                        0.8,
                        2
                    )

        if self.smile_triggered:

            self.draw_text(
                frame,
                "SMILE DETECTED",
                420,
                0.75,
                2
            )

        # -------------------------------------------------
        # FINISH
        # -------------------------------------------------

        if test_elapsed >= SMILE_DURATION:

            if not self.smile_triggered:

                self.smile_score = 90

            self.change_state(
                "ANALYZING"
            )

    # =====================================================
    # ANALYSIS
    # =====================================================

    def analyzing(self, frame):

        elapsed = (
            time.time()
            -
            self.state_start
        )

        self.draw_text(
            frame,
            "ANALYZING HUMANITY",
            250,
            1.3,
            3
        )

        dots = "." * (
            int(elapsed * 3) % 4
        )

        self.draw_text(
            frame,
            dots,
            330,
            1.0,
            2
        )

        if elapsed >= 3:

            self.final_score = (
                self.movement_score * 0.40
                +
                self.blink_score * 0.30
                +
                self.smile_score * 0.30
            )

            self.change_state(
                "RESULT"
            )

    # =====================================================
    # RESULT SCREEN
    # =====================================================

    def result(self, frame):

        npc_class = get_npc_class(
            self.final_score
        )

        self.draw_text(
            frame,
            "HUMANITY REPORT",
            100,
            1.3,
            3
        )

        self.draw_text(
            frame,
            f"NPC PROBABILITY: {self.final_score:.1f}%",
            190,
            0.95,
            2
        )

        self.draw_text(
            frame,
            npc_class,
            290,
            1.0,
            3
        )

        self.draw_text(
            frame,
            f"MOVEMENT: {self.movement_score:.1f}",
            370,
            0.6,
            2
        )

        self.draw_text(
            frame,
            f"BLINK: {self.blink_score:.1f}",
            415,
            0.6,
            2
        )

        self.draw_text(
            frame,
            f"REACTION: {self.smile_score:.1f}",
            460,
            0.6,
            2
        )

        self.draw_text(
            frame,
            "PRESS R TO TEST AGAIN     Q TO QUIT",
            550,
            0.55,
            1
        )

    # =====================================================
    # MAIN LOOP
    # =====================================================

    def run(self):

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
            # DARK OVERLAY
            # -------------------------------------------------

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (
                    frame.shape[1],
                    frame.shape[0]
                ),
                (0, 0, 0),
                -1
            )

            frame = cv2.addWeighted(
                frame,
                0.45,
                overlay,
                0.55,
                0
            )

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
                "NPC DETECTOR™",
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