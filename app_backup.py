import cv2
import mediapipe as mp
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# MEDIAPIPE FACE DETECTOR
# =========================

base_options = python.BaseOptions(
    model_asset_path="face_detector.tflite"
)

options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)

detector = vision.FaceDetector.create_from_options(options)


# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(0)

print("NPC Detector™ starting...")
print("Press Q to quit.")


previous_center = None
movement_total = 0
frames = 0


# =========================
# MAIN LOOP
# =========================

while True:

    success, frame = cap.read()

    if not success:
        print("Could not access webcam.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # OpenCV BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect faces
    result = detector.detect(mp_image)


    # =========================
    # FACE FOUND
    # =========================

    if result.detections:

        # Select largest face
        detection = max(
            result.detections,
            key=lambda d:
                d.bounding_box.width *
                d.bounding_box.height
        )

        box = detection.bounding_box

        x = box.origin_x
        y = box.origin_y
        width = box.width
        height = box.height


        # Face center
        center_x = x + width // 2
        center_y = y + height // 2


        # =========================
        # MOVEMENT DETECTION
        # =========================

        if previous_center is not None:

            distance = math.sqrt(
                (center_x - previous_center[0]) ** 2 +
                (center_y - previous_center[1]) ** 2
            )

            movement_total += distance

        previous_center = (
            center_x,
            center_y
        )

        frames += 1


        # =========================
        # DRAW FACE
        # =========================

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )


        # =========================
        # MOVEMENT SCORE
        # =========================

        average_movement = (
            movement_total / frames
            if frames > 1
            else 0
        )

        npc_movement_score = max(
            0,
            min(
                100,
                100 - average_movement * 2
            )
        )


        # =========================
        # VERDICT
        # =========================

        if npc_movement_score > 80:

            verdict = "IDLE NPC DETECTED"

        elif npc_movement_score > 60:

            verdict = "SUSPICIOUSLY STILL"

        elif npc_movement_score > 40:

            verdict = "NORMAL HUMAN"

        else:

            verdict = "VERY HUMAN"


        # =========================
        # DISPLAY
        # =========================

        cv2.putText(
            frame,
            "HUMAN DETECTED",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"NPC SCORE: {npc_movement_score:.0f}%",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            verdict,
            (30, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


    # =========================
    # NO FACE
    # =========================

    else:

        previous_center = None

        cv2.putText(
            frame,
            "NO HUMAN DETECTED",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # =========================
    # SHOW
    # =========================

    cv2.imshow(
        "NPC Detector™",
        frame
    )


    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEANUP
# =========================

cap.release()
detector.close()
cv2.destroyAllWindows()

print("NPC Detector™ stopped.")