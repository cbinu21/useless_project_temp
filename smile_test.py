import cv2
import mediapipe as mp
import time
import random
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# MEDIAPIPE FACE LANDMARKER
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

if not cap.isOpened():
    print("Could not access webcam.")
    exit()

print("======================================")
print("       NPC HUMANITY TEST™")
print("       SMILE CHALLENGE")
print("======================================")
print("Press Q to quit.")


# ==========================================
# STATES
# ==========================================

READY = 0
SMILE = 1
RESULT = 2

state = READY

state_start = time.time()

wait_time = random.uniform(2, 5)

reaction_start = None
reaction_time = None

baseline_smile = None

# Prevent accidental repeated detection
smile_detected = False


# ==========================================
# SMILE MEASUREMENT
# ==========================================

def smile_measure(landmarks):

    # Mouth corners
    left_corner = landmarks[61]
    right_corner = landmarks[291]

    # Upper/lower lip
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]

    # Face width reference
    left_face = landmarks[234]
    right_face = landmarks[454]

    mouth_width = math.dist(
        (left_corner.x, left_corner.y),
        (right_corner.x, right_corner.y)
    )

    mouth_height = math.dist(
        (upper_lip.x, upper_lip.y),
        (lower_lip.x, lower_lip.y)
    )

    face_width = math.dist(
        (left_face.x, left_face.y),
        (right_face.x, right_face.y)
    )

    if face_width == 0:
        return 0

    # Normalized mouth width
    width_ratio = mouth_width / face_width

    # Mouth opening
    height_ratio = mouth_height / face_width

    # Combined smile indicator
    score = width_ratio + height_ratio * 0.5

    return score


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read webcam.")
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
        time.time() * 1000
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # ======================================
    # FACE DETECTED
    # ======================================

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        current_smile = smile_measure(
            landmarks
        )


        # ==================================
        # READY PHASE
        # ==================================

        if state == READY:

            cv2.putText(
                frame,
                "HUMANITY TEST",
                (110, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (255, 255, 255),
                3
            )

            cv2.putText(
                frame,
                "GET READY...",
                (145, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 255),
                3
            )

            cv2.putText(
                frame,
                "DO NOT SMILE YET",
                (135, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            # Capture neutral face during preparation
            if baseline_smile is None:

                baseline_smile = current_smile

            else:

                # Smooth baseline
                baseline_smile = (
                    baseline_smile * 0.95
                    + current_smile * 0.05
                )


            # Random delay
            if time.time() - state_start >= wait_time:

                state = SMILE

                reaction_start = time.time()

                smile_detected = False


        # ==================================
        # SMILE PHASE
        # ==================================

        elif state == SMILE:

            cv2.putText(
                frame,
                "SMILE!",
                (150, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.2,
                (0, 255, 0),
                5
            )

            cv2.putText(
                frame,
                "SHOW US THOSE TEETH",
                (90, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


            # Smile threshold
            threshold = baseline_smile * 1.18


            # Detect significant smile
            if (
                current_smile > threshold
                and not smile_detected
            ):

                reaction_time = (
                    time.time()
                    - reaction_start
                )

                smile_detected = True

                state = RESULT

                state_start = time.time()


        # ==================================
        # RESULT
        # ==================================

        elif state == RESULT:

            cv2.putText(
                frame,
                "SMILE DETECTED!",
                (80, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"REACTION: {reaction_time:.3f} SEC",
                (70, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )


            # Verdict
            if reaction_time < 0.4:

                verdict = "SUSPICIOUSLY FAST"

            elif reaction_time < 0.8:

                verdict = "HUMAN REFLEXES DETECTED"

            elif reaction_time < 1.5:

                verdict = "NORMAL HUMAN"

            elif reaction_time < 3:

                verdict = "NPC-LIKE DELAY"

            else:

                verdict = "YOUR ANIMATION LOADED SLOWLY"


            cv2.putText(
                frame,
                verdict,
                (55, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Press R to test again",
                (105, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            # Restart
            if cv2.waitKey(1) & 0xFF == ord("r"):

                state = READY

                state_start = time.time()

                wait_time = random.uniform(2, 5)

                reaction_start = None
                reaction_time = None

                baseline_smile = None
                smile_detected = False


    else:

        cv2.putText(
            frame,
            "FACE NOT DETECTED",
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ======================================
    # CAMERA WINDOW
    # ======================================

    cv2.imshow(
        "NPC Detector™ - Humanity Test",
        frame
    )


    # Quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

landmarker.close()

cv2.destroyAllWindows()

print("Humanity test complete.")