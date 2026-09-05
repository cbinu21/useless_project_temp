import cv2
import mediapipe as mp
import math
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# MEDIAPIPE FACE LANDMARKS
# =========================

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


# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(0)

blink_count = 0
eyes_closed = False

start_time = time.time()


# =========================
# EYE ASPECT RATIO
# =========================

def eye_aspect_ratio(landmarks, points, width, height):

    coords = []

    for index in points:
        x = landmarks[index].x * width
        y = landmarks[index].y * height
        coords.append((x, y))

    # Vertical distances
    vertical_1 = math.dist(coords[1], coords[5])
    vertical_2 = math.dist(coords[2], coords[4])

    # Horizontal distance
    horizontal = math.dist(coords[0], coords[3])

    return (vertical_1 + vertical_2) / (2 * horizontal)


# MediaPipe eye landmark points
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# =========================
# MAIN LOOP
# =========================

print("BLINK TEST STARTED")
print("Look at the camera and blink.")
print("Press Q to quit.")

while True:

    success, frame = cap.read()

    if not success:
        print("Could not access webcam.")
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int((time.time() - start_time) * 1000)

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        left_ear = eye_aspect_ratio(
            landmarks,
            LEFT_EYE,
            width,
            height
        )

        right_ear = eye_aspect_ratio(
            landmarks,
            RIGHT_EYE,
            width,
            height
        )

        ear = (left_ear + right_ear) / 2


        # =========================
        # BLINK DETECTION
        # =========================

        if ear < 0.21:

            if not eyes_closed:
                blink_count += 1
                eyes_closed = True

        else:

            eyes_closed = False


        # =========================
        # DISPLAY
        # =========================

        cv2.putText(
            frame,
            f"BLINKS: {blink_count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"EYE RATIO: {ear:.2f}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if eyes_closed:

            cv2.putText(
                frame,
                "EYES CLOSED",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "EYES OPEN",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


    cv2.imshow("NPC Detector - Blink Test", frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
landmarker.close()
cv2.destroyAllWindows()

print(f"Total blinks detected: {blink_count}")