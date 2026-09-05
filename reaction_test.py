import cv2
import time
import random

# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(0)

print("================================")
print("      NPC HUMANITY TEST™")
print("================================")
print("Get ready...")
print("Press Q to quit.")


# =========================
# STATES
# =========================

WAITING = 0
READY = 1
GO = 2
RESULT = 3

state = WAITING

state_start = time.time()
reaction_start = None
reaction_time = None

# Random delay before GO
wait_time = random.uniform(2, 5)


# =========================
# MAIN LOOP
# =========================

while True:

    success, frame = cap.read()

    if not success:
        print("Could not access webcam.")
        break

    frame = cv2.flip(frame, 1)

    now = time.time()


    # =========================
    # WAITING
    # =========================

    if state == WAITING:

        cv2.putText(
            frame,
            "HUMANITY TEST",
            (120, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            "GET READY...",
            (150, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3
        )

        if now - state_start >= wait_time:

            state = GO
            reaction_start = time.time()


    # =========================
    # GO
    # =========================

    elif state == GO:

        cv2.putText(
            frame,
            "GO!",
            (250, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            5
        )

        # For v1, press SPACE as the human response
        cv2.putText(
            frame,
            "PRESS SPACE!",
            (130, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )


        # SPACE = response
        if cv2.waitKey(1) & 0xFF == ord(' '):

            reaction_time = time.time() - reaction_start

            state = RESULT
            state_start = time.time()


    # =========================
    # RESULT
    # =========================

    elif state == RESULT:

        cv2.putText(
            frame,
            f"REACTION TIME: {reaction_time:.3f} SEC",
            (60, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )


        if reaction_time < 0.25:

            verdict = "SUSPICIOUSLY FAST"

        elif reaction_time < 0.6:

            verdict = "HUMAN REFLEXES DETECTED"

        elif reaction_time < 1.2:

            verdict = "NORMAL HUMAN"

        else:

            verdict = "NPC-LIKE RESPONSE"


        cv2.putText(
            frame,
            verdict,
            (90, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            "Press R to test again",
            (110, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        if cv2.waitKey(1) & 0xFF == ord('r'):

            state = WAITING
            state_start = time.time()
            wait_time = random.uniform(2, 5)


    # =========================
    # SHOW
    # =========================

    cv2.imshow(
        "NPC Humanity Test™",
        frame
    )


    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

print("Humanity test finished.")