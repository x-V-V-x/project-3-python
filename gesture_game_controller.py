import cv2
import mediapipe as mp
import pyautogui

# Setup webcam and mediapipe
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands()
draw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip horizontally
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            landmarks = hand.landmark

            # Get tip of index, middle, ring, pinky fingers
            finger_tips = [8, 12, 16, 20]
            folded = 0

            for tip in finger_tips:
                if landmarks[tip].y > landmarks[tip - 2].y:
                    folded += 1

            # If all 4 fingers are folded → FIST
            if folded == 4:
                print("Fist detected — jumping!")
                pyautogui.press('space')  # Press space key

            draw.draw_landmarks(
                frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Game Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
