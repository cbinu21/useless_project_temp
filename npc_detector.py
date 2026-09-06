import cv2
import time
import math
import random
import mediapipe as mp

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# NPC DETECTOR™
# =========================================================

MOVEMENT_DURATION = 6
BLINK_DURATION = 8
SMILE_CALIBRATION = 2
SMILE_DURATION = 5

BLINK_THRESHOLD = 0.21


# =========================================================
# MEDIAPIPE
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

LEFT_EYE = [
    33, 160, 158,
    133, 153, 144
]

RIGHT_EYE = [
    362, 385, 387,
    263, 373, 380
]


# =========================================================
# UTILITY
# =========================================================

def distance(a, b):

    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


def eye_aspect_ratio(
    landmarks,
    eye
):

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
        vertical_1 +
        vertical_2
    ) / (2.0 * horizontal)


def clamp(
    value,
    minimum,
    maximum
):

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


# =========================================================
# NPC CLASSIFICATION
# =========================================================

def get_npc_class(score):

    if score >= 85:
        return "BACKGROUND VILLAGER"

    if score >= 70:
        return "SHOPKEEPER NPC"

    if score >= 55:
        return "QUEST NPC"

    if score >= 35:
        return "PLAYABLE CHARACTER"

    return "CHAOTIC PLAYER"


# =========================================================
# NPC BEHAVIOUR
# =========================================================

def get_behaviour(score):

    if score >= 85:

        options = [
            "Likely to stand in the same place forever.",
            "May repeat the same dialogue every 30 seconds.",
            "Background character behaviour detected.",
            "Subject appears to have no main quest."
        ]

    elif score >= 70:

        options = [
            "Probably guarding an important doorway.",
            "May sell suspiciously expensive potions.",
            "Will probably say the same thing tomorrow.",
            "Subject appears to know exactly one useful sentence."
        ]

    elif score >= 55:

        options = [
            "Possibly hiding a side quest.",
            "Subject may know something important.",
            "Quest-giving behaviour suspected.",
            "Probably waiting for someone to talk to them."
        ]

    elif score >= 35:

        options = [
            "Free will detected. Unfortunately.",
            "Subject appears capable of making choices.",
            "Main-character tendencies detected.",
            "Potentially dangerous levels of autonomy."
        ]

    else:

        options = [
            "EXTREME FREE WILL DETECTED.",
            "Subject refuses to follow game logic.",
            "Definitely not an NPC.",
            "Developer has lost control of this character."
        ]

    return random.choice(options)


# =========================================================
# NPC DIALOGUE
# =========================================================

def get_dialogue(score):

    if score >= 85:
        return "The weather is strange today."

    if score >= 70:
        return "Welcome, traveller. Need anything?"

    if score >= 55:
        return "I have a quest for you."

    if score >= 35:
        return "You look familiar..."

    return "STOP MOVING LIKE THAT."


# =========================================================
# APPLICATION
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
        # STATE
        # -------------------------------------------------

        self.running = True

        self.state = "BOOT"

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
        # MOVEMENT
        # -------------------------------------------------

        self.position_history = deque(
            maxlen=60
        )

        self.movement_values = []

        # -------------------------------------------------
        # BLINK
        # -------------------------------------------------

        self.blink_count = 0

        self.eye_closed = False

        # -------------------------------------------------
        # SMILE
        # -------------------------------------------------

        self.smile_values = []

        self.smile_baseline = None

        self.smile_triggered = False

        self.smile_start = None

        self.smile_reaction = None

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        self.behaviour = ""

        self.dialogue = ""

        # -------------------------------------------------
        # WINDOW
        # -------------------------------------------------

        self.window_name = "NPC DETECTOR"

    # =====================================================
    # STATE CHANGE
    # =====================================================

    def change_state(self, new_state):

        self.state = new_state

        self.state_start = time.time()

        # MOVEMENT

        if new_state == "MOVEMENT":

            self.position_history.clear()

            self.movement_values.clear()

            self.movement_score = 0

        # BLINK

        elif new_state == "BLINK":

            self.blink_count = 0

            self.eye_closed = False

            self.blink_score = 0

        # SMILE

        elif new_state == "SMILE":

            self.smile_values.clear()

            self.smile_baseline = None

            self.smile_triggered = False

            self.smile_start = None

            self.smile_reaction = None

            self.smile_score = 0

        # WELCOME

        elif new_state == "WELCOME":

            self.final_score = 0

    # =====================================================
    # KEYBOARD
    # =====================================================

    def handle_key(self, key):

        if key == ord("q"):

            self.running = False

        elif key == ord("r"):

            self.change_state(
                "WELCOME"
            )

        elif key == 32:

            if self.state == "WELCOME":

                self.change_state(
                    "MOVEMENT"
                )

    # =====================================================
    # TEXT
    # =====================================================

    def draw_text(
        self,
        frame,
        message,
        x,
        y,
        scale=1.0,
        thickness=2
    ):

        cv2.putText(
            frame,
            message,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    # =====================================================
    # CENTERED TEXT
    # =====================================================

    def centered_text(
        self,
        frame,
        message,
        y,
        scale=1.0,
        thickness=2
    ):

        font = cv2.FONT_HERSHEY_SIMPLEX

        size, _ = cv2.getTextSize(
            message,
            font,
            scale,
            thickness
        )

        x = (
            frame.shape[1]
            -
            size[0]
        ) // 2

        self.draw_text(
            frame,
            message,
            x,
            y,
            scale,
            thickness
        )

    # =====================================================
    # DARK OVERLAY
    # =====================================================

    def darken(
        self,
        frame,
        amount=0.35
    ):

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

        frame[:] = cv2.addWeighted(
            overlay,
            amount,
            frame,
            1 - amount,
            0
        )

    # =====================================================
    # HUD BAR
    # =====================================================

    def hud_bar(
        self,
        frame,
        y1,
        y2,
        alpha=0.40
    ):

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, y1),
            (
                frame.shape[1],
                y2
            ),
            (0, 0, 0),
            -1
        )

        frame[:] = cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0
        )

    # =====================================================
    # PROGRESS BAR
    # =====================================================

    def progress_bar(
        self,
        frame,
        elapsed,
        duration,
        y
    ):

        width = 700

        height = 10

        x = (
            frame.shape[1]
            -
            width
        ) // 2

        progress = clamp(
            elapsed / duration,
            0,
            1
        )

        cv2.rectangle(
            frame,
            (x, y),
            (
                x + width,
                y + height
            ),
            (70, 70, 70),
            -1
        )

        cv2.rectangle(
            frame,
            (x, y),
            (
                x +
                int(
                    width *
                    progress
                ),
                y + height
            ),
            (255, 255, 255),
            -1
        )

    # =====================================================
    # TOP BAR
    # =====================================================

    def top_bar(
        self,
        frame,
        status
    ):

        self.hud_bar(
            frame,
            0,
            70,
            0.45
        )

        self.draw_text(
            frame,
            "NPC DETECTOR™",
            30,
            43,
            0.65,
            2
        )

        self.draw_text(
            frame,
            status,
            frame.shape[1] - 280,
            43,
            0.55,
            2
        )

    # =====================================================
    # BOOT SCREEN
    # =====================================================

    def boot(
        self,
        frame
    ):

        elapsed = (
            time.time()
            -
            self.state_start
        )

        self.darken(
            frame,
            0.72
        )

        self.centered_text(
            frame,
            "NPC DETECTOR™",
            230,
            2.0,
            4
        )

        self.centered_text(
            frame,
            "HUMANITY VERIFICATION SYSTEM",
            290,
            0.65,
            2
        )

        messages = [
            "INITIALIZING NEURAL OBSERVATION...",
            "LOADING HUMAN BEHAVIOUR DATABASE...",
            "CALIBRATING SUSPICION ENGINE...",
            "SEARCHING FOR FREE WILL...",
            "SYSTEM READY."
        ]

        index = min(
            int(elapsed * 1.4),
            len(messages) - 1
        )

        self.centered_text(
            frame,
            messages[index],
            390,
            0.6,
            2
        )

        self.progress_bar(
            frame,
            elapsed,
            3.8,
            430
        )

        if elapsed >= 4:

            self.change_state(
                "WELCOME"
            )

    # =====================================================
    # WELCOME
    # =====================================================

    def welcome(
        self,
        frame
    ):

        self.darken(
            frame,
            0.30
        )

        self.top_bar(
            frame,
            "SYSTEM ONLINE"
        )

        self.centered_text(
            frame,
            "NPC DETECTOR™",
            210,
            2.0,
            4
        )

        self.centered_text(
            frame,
            "HUMANITY VERIFICATION SYSTEM",
            270,
            0.75,
            2
        )

        self.centered_text(
            frame,
            "We have serious concerns about your humanity.",
            340,
            0.65,
            2
        )

        # Start button

        x1 = 350

        y1 = 390

        x2 = frame.shape[1] - 350

        y2 = 460

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        self.centered_text(
            frame,
            "[ SPACE ]  BEGIN HUMANITY TEST",
            435,
            0.7,
            2
        )

        self.centered_text(
            frame,
            "R  RESTART       Q  QUIT",
            535,
            0.55,
            1
        )

    # =====================================================
    # MOVEMENT
    # =====================================================

    def movement_test(
        self,
        frame,
        landmarks
    ):

        elapsed = (
            time.time()
            -
            self.state_start
        )

        remaining = max(
            0,
            MOVEMENT_DURATION
            -
            int(elapsed)
        )

        self.top_bar(
            frame,
            "TEST 01 / 03"
        )

        self.centered_text(
            frame,
            "MOVEMENT ANALYSIS",
            130,
            1.2,
            3
        )

        self.centered_text(
            frame,
            f"MOVE YOUR HEAD NATURALLY     {remaining}",
            180,
            0.65,
            2
        )

        self.progress_bar(
            frame,
            elapsed,
            MOVEMENT_DURATION,
            210
        )

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

                old_x, old_y = (
                    self.position_history[-2]
                )

                movement = math.sqrt(
                    (x - old_x) ** 2
                    +
                    (y - old_y) ** 2
                )

                self.movement_values.append(
                    movement
                )

            h, w = frame.shape[:2]

            cx = int(
                nose.x * w
            )

            cy = int(
                nose.y * h
            )

            cv2.circle(
                frame,
                (cx, cy),
                12,
                (255, 255, 255),
                2
            )

            cv2.circle(
                frame,
                (cx, cy),
                3,
                (255, 255, 255),
                -1
            )

            self.hud_bar(
                frame,
                frame.shape[0] - 85,
                frame.shape[0],
                0.45
            )

            self.centered_text(
                frame,
                "● FACE TRACKED  |  OBSERVING MOVEMENT",
                frame.shape[0] - 42,
                0.55,
                2
            )

        else:

            self.hud_bar(
                frame,
                frame.shape[0] - 85,
                frame.shape[0],
                0.45
            )

            self.centered_text(
                frame,
                "● FACE NOT DETECTED",
                frame.shape[0] - 42,
                0.55,
                2
            )

        if elapsed >= MOVEMENT_DURATION:

            if self.movement_values:

                average = (
                    sum(
                        self.movement_values
                    )
                    /
                    len(
                        self.movement_values
                    )
                )

                self.movement_score = clamp(
                    100 -
                    average * 300,
                    0,
                    100
                )

            else:

                self.movement_score = 100

            self.change_state(
                "BLINK"
            )

    # =====================================================
    # BLINK
    # =====================================================

    def blink_test(
        self,
        frame,
        landmarks
    ):

        elapsed = (
            time.time()
            -
            self.state_start
        )

        remaining = max(
            0,
            BLINK_DURATION
            -
            int(elapsed)
        )

        self.top_bar(
            frame,
            "TEST 02 / 03"
        )

        self.centered_text(
            frame,
            "BLINK ANALYSIS",
            130,
            1.2,
            3
        )

        self.centered_text(
            frame,
            f"BLINK NATURALLY     {remaining}",
            180,
            0.7,
            2
        )

        self.progress_bar(
            frame,
            elapsed,
            BLINK_DURATION,
            210
        )

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
                left_ear +
                right_ear
            ) / 2

            if ear < BLINK_THRESHOLD:

                self.eye_closed = True

                status = "EYES CLOSED"

            else:

                if self.eye_closed:

                    self.blink_count += 1

                self.eye_closed = False

                status = "EYES OPEN"

            self.hud_bar(
                frame,
                270,
                505,
                0.32
            )

            self.centered_text(
                frame,
                status,
                350,
                1.1,
                3
            )

            self.centered_text(
                frame,
                f"BLINKS DETECTED: {self.blink_count}",
                415,
                0.9,
                2
            )

            self.centered_text(
                frame,
                f"EYE RATIO: {ear:.2f}",
                460,
                0.60,
                2
            )

        else:

            self.centered_text(
                frame,
                "FACE NOT DETECTED",
                350,
                0.8,
                2
            )

        if elapsed >= BLINK_DURATION:

            if self.blink_count == 0:

                self.blink_score = 95

            elif self.blink_count == 1:

                self.blink_score = 75

            elif self.blink_count == 2:

                self.blink_score = 45

            elif self.blink_count <= 4:

                self.blink_score = 20

            else:

                self.blink_score = 60

            self.change_state(
                "SMILE"
            )

    # =====================================================
    # SMILE
    # =====================================================

    def smile_test(
        self,
        frame,
        landmarks
    ):

        elapsed = (
            time.time()
            -
            self.state_start
        )

        # Calibration

        if elapsed < SMILE_CALIBRATION:

            self.top_bar(
                frame,
                "TEST 03 / 03"
            )

            self.centered_text(
                frame,
                "REACTION CALIBRATION",
                170,
                1.1,
                3
            )

            self.centered_text(
                frame,
                "DO NOT SMILE",
                245,
                0.8,
                2
            )

            self.progress_bar(
                frame,
                elapsed,
                SMILE_CALIBRATION,
                285
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
                        mouth_width /
                        face_width
                    )

                    self.smile_values.append(
                        ratio
                    )

            return

        # Baseline

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

        test_elapsed = (
            time.time()
            -
            self.smile_start
        )

        self.top_bar(
            frame,
            "TEST 03 / 03"
        )

        self.centered_text(
            frame,
            "REACTION ANALYSIS",
            130,
            1.2,
            3
        )

        self.centered_text(
            frame,
            "SMILE!",
            220,
            1.8,
            4
        )

        self.centered_text(
            frame,
            "HOW FAST CAN YOU LOOK HUMAN?",
            275,
            0.65,
            2
        )

        self.progress_bar(
            frame,
            test_elapsed,
            SMILE_DURATION,
            310
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
                    mouth_width /
                    face_width
                )

                threshold = (
                    self.smile_baseline
                    * 1.18
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

        if self.smile_triggered:

            self.hud_bar(
                frame,
                350,
                480,
                0.30
            )

            self.centered_text(
                frame,
                "✓ SMILE DETECTED",
                405,
                0.9,
                3
            )

            self.centered_text(
                frame,
                f"REACTION TIME: {self.smile_reaction:.2f}s",
                450,
                0.6,
                2
            )

        else:

            self.centered_text(
                frame,
                "WAITING FOR HUMAN EXPRESSION...",
                405,
                0.6,
                1
            )

        if test_elapsed >= SMILE_DURATION:

            if not self.smile_triggered:

                self.smile_score = 90

            self.change_state(
                "ANALYZING"
            )

    # =====================================================
    # ANALYSIS
    # =====================================================

    def analyzing(
        self,
        frame
    ):

        elapsed = (
            time.time()
            -
            self.state_start
        )

        self.darken(
            frame,
            0.78
        )

        self.top_bar(
            frame,
            "PROCESSING"
        )

        self.centered_text(
            frame,
            "HUMANITY ANALYSIS",
            180,
            1.4,
            3
        )

        self.centered_text(
            frame,
            "PROCESSING BEHAVIOURAL EVIDENCE",
            235,
            0.65,
            2
        )

        self.progress_bar(
            frame,
            elapsed,
            4,
            285
        )

        steps = [

            "ANALYSING MOVEMENT ........ COMPLETE",

            "ANALYSING OCULAR BEHAVIOUR ... COMPLETE",

            "ANALYSING REACTION ........... COMPLETE",

            "CALCULATING NPC PROBABILITY ...",

            "GENERATING SUBJECT PROFILE ..."
        ]

        index = min(
            int(
                elapsed * 1.25
            ),
            len(steps) - 1
        )

        self.centered_text(
            frame,
            steps[index],
            365,
            0.60,
            2
        )

        if elapsed >= 4:

            self.final_score = (

                self.movement_score
                * 0.40

                +

                self.blink_score
                * 0.30

                +

                self.smile_score
                * 0.30
            )

            self.behaviour = get_behaviour(
                self.final_score
            )

            self.dialogue = get_dialogue(
                self.final_score
            )

            self.change_state(
                "RESULT"
            )

    # =====================================================
    # RESULT
    # =====================================================

    def result(
        self,
        frame
    ):

        self.darken(
            frame,
            0.72
        )

        self.top_bar(
            frame,
            "ANALYSIS COMPLETE"
        )

        npc_class = get_npc_class(
            self.final_score
        )

        self.centered_text(
            frame,
            "HUMANITY REPORT",
            115,
            1.25,
            3
        )

        self.centered_text(
            frame,
            "NPC PROBABILITY",
            175,
            0.60,
            2
        )

        self.centered_text(
            frame,
            f"{self.final_score:.1f}%",
            265,
            2.25,
            4
        )

        self.centered_text(
            frame,
            npc_class,
            330,
            1.0,
            3
        )

        # Divider

        cv2.line(
            frame,
            (250, 360),
            (
                frame.shape[1] - 250,
                360
            ),
            (150, 150, 150),
            1
        )

        # Scores

        self.centered_text(
            frame,
            f"MOVEMENT   {self.movement_score:.0f}"
            f"    |    "
            f"BLINK   {self.blink_score:.0f}"
            f"    |    "
            f"REACTION   {self.smile_score:.0f}",
            410,
            0.55,
            2
        )

        # Behaviour

        self.centered_text(
            frame,
            "BEHAVIOURAL ASSESSMENT",
            455,
            0.50,
            1
        )

        # Break behaviour into lines if needed

        behaviour = self.behaviour

        if len(behaviour) > 55:

            split = behaviour.rfind(
                " ",
                0,
                55
            )

            line1 = behaviour[:split]

            line2 = behaviour[split + 1:]

            self.centered_text(
                frame,
                line1,
                490,
                0.58,
                2
            )

            self.centered_text(
                frame,
                line2,
                520,
                0.58,
                2
            )

        else:

            self.centered_text(
                frame,
                behaviour,
                495,
                0.58,
                2
            )

        # Dialogue

        self.centered_text(
            frame,
            '"' + self.dialogue + '"',
            555,
            0.58,
            2
        )

        # Controls

        self.centered_text(
            frame,
            "R  TEST AGAIN       Q  EXIT",
            610,
            0.55,
            2
        )

    # =====================================================
    # MAIN LOOP
    # =====================================================

    def run(self):

        # -------------------------------------------------
        # FULLSCREEN
        # -------------------------------------------------

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.setWindowProperty(
            self.window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

        # -------------------------------------------------
        # LOOP
        # -------------------------------------------------

        while self.running:

            success, frame = (
                self.cap.read()
            )

            if not success:

                continue

            # Mirror

            frame = cv2.flip(
                frame,
                1
            )

            # -------------------------------------------------
            # MEDIAPIPE
            # -------------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

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
            # STATE
            # -------------------------------------------------

            if self.state == "BOOT":

                self.boot(frame)

            elif self.state == "WELCOME":

                self.welcome(frame)

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

                self.analyzing(frame)

            elif self.state == "RESULT":

                self.result(frame)

            # -------------------------------------------------
            # SHOW
            # -------------------------------------------------

            cv2.imshow(
                self.window_name,
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

        # -------------------------------------------------
        # CLEANUP
        # -------------------------------------------------

        self.cap.release()

        cv2.destroyAllWindows()


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    app = NPCDetector()

    app.run()