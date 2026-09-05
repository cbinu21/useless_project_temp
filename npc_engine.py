import cv2
import mediapipe as mp
import math
import time
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# MEDIAPIPE
# ==========================================

base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,
    running_mode=vision.RunningMode.VIDEO
)

landmarker = vision.FaceLandmarker.create_from_options(options)


# ==========================================
# WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

print("================================")
print("       NPC DETECTOR™")
print("================================")
print("Analyzing human behavior...")
print("Press Q to quit.")


# ==========================================
# VARIABLES
# ==========================================

previous_center = None

movement_history = deque(maxlen=60)

blink_count = 0
eyes_closed = False

start_time = time.time()


# ==========================================
# EYE ASPECT RATIO
# ==========================================

def eye_aspect_ratio(landmarks, points):

    coords = []

    for index in points:
        x = landmarks[index].x
        y = landmarks[index].y
        coords.append((x, y))

    vertical_1 = math.dist(coords[1], coords[5])
    vertical_2 = math.dist(coords[2], coords[4])

    horizontal = math.dist(coords[0], coords[3])

    return (vertical_1 + vertical_2) / (2 * horizontal)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        print("Could not access webcam.")
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int(
        (time.time() - start_time) * 1000
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # ======================================
    # FACE FOUND
    # ======================================

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]


        # ==================================
        # FACE CENTER
        # ==================================

        nose = landmarks[1]

        center_x = nose.x * width
        center_y = nose.y * height


        # ==================================
        # MOVEMENT
        # ==================================

        if previous_center is not None:

            movement = math.dist(
                (center_x, center_y),
                previous_center
            )

            movement_history.append(movement)

        previous_center = (
            center_x,
            center_y
        )


        # ==================================
        # BLINK
        # ==================================

        left_ear = eye_aspect_ratio(
            landmarks,
            LEFT_EYE
        )

        right_ear = eye_aspect_ratio(
            landmarks,
            RIGHT_EYE
        )

        ear = (left_ear + right_ear) / 2


        if ear < 0.21:

            if not eyes_closed:

                blink_count += 1
                eyes_closed = True

        else:

            eyes_closed = False


        # ==================================
        # MOVEMENT SCORE
        # ==================================

        if len(movement_history) > 5:

            average_movement = (
                sum(movement_history)
                / len(movement_history)
            )

        else:

            average_movement = 0


        # Less movement = more NPC-like

        movement_npc = max(
            0,
            min(
                100,
                100 - average_movement * 3
            )
        )


        # ==================================
        # BLINK SCORE
        # ==================================

        elapsed_minutes = (
            time.time() - start_time
        ) / 60

        if elapsed_minutes > 0:

            blink_rate = (
                blink_count
                / elapsed_minutes
            )

        else:

            blink_rate = 0


        # Normal human blink rate ≈ 10–20/min

        if blink_rate < 3:

            blink_npc = 90

        elif blink_rate < 7:

            blink_npc = 65

        elif blink_rate <= 25:

            blink_npc = 25

        else:

            blink_npc = 60


        # ==================================
        # FINAL NPC SCORE
        # ==================================

        npc_score = (
            movement_npc * 0.65
            + blink_npc * 0.35
        )

        npc_score = max(
            0,
            min(100, npc_score)
        )


        # ==================================
        # NPC TYPE
        # ==================================

        if npc_score >= 85:

            npc_type = "BACKGROUND VILLAGER"

        elif npc_score >= 70:

            npc_type = "SHOPKEEPER NPC"

        elif npc_score >= 55:

            npc_type = "QUEST NPC"

        elif npc_score >= 35:

            npc_type = "PLAYABLE CHARACTER"

        else:

            npc_type = "CHAOTIC PLAYER"


        # ==================================
        # DISPLAY
        # ==================================

        cv2.putText(
            frame,
            "NPC DETECTOR™",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"NPC SCORE: {npc_score:.0f}%",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"BLINKS: {blink_count}",
            (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"BLINK RATE: {blink_rate:.1f}/min",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            npc_type,
            (30, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )


    else:

        previous_center = None

        cv2.putText(
            frame,
            "NO HUMAN DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ======================================
    # SHOW
    # ======================================

    cv2.imshow(
        "NPC Detector™",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print("NPC Detector™ stopped.")