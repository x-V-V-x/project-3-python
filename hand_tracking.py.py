import cv2
import mediapipe as mp

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # You can detect more hands
mp_draw = mp.solutions.drawing_utils

while True:
    success, frame = cap.read()

    # Convert the image to RGB (MediaPipe requires this format)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB image to detect hands
    results = hands.process(img_rgb)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the original BGR image
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

    # Show the output
    cv2.imshow("Hand Tracking", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
