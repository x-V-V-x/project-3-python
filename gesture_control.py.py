import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip to make it mirror-like
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    fingers = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            h, w, c = frame.shape

            # Get all landmarks
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Check fingers
            # Thumb (compare x values because it moves sideways)
            if lm_list[4][1] > lm_list[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other fingers (compare y values)
            for tip_id in tip_ids[1:]:
                if lm_list[tip_id][2] < lm_list[tip_id - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Gesture logic
            gesture = ""
            if fingers == [0, 0, 0, 0, 0]:
                gesture = "Fist 👊"
            elif fingers == [1, 1, 1, 1, 1]:
                gesture = "Open Hand 🖐"
            elif fingers == [1, 0, 0, 0, 0]:
                gesture = "Thumbs Up 👍"
            elif fingers == [0, 1, 1, 0, 0]:
                gesture = "Victory ✌️"
            else:
                gesture = f"{fingers.count(1)} Fingers Up"

            # Draw on hand and display gesture
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, gesture, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
