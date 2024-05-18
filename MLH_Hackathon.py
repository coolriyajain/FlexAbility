import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
import cv2
import mediapipe as mp
import numpy as np
import random
import time
print("Welcome to the Gesture Recognition Game!")
print("In this game, you will be prompted to make different hand gestures using your webcam. The game will recognize your gestures and award points for successfully holding the requested gesture.")

print("Gestures:")
print("The game recognizes the following hand gestures:")
print("- Peace Sign (Two extended fingers)")
print("- Rock and Roll Sign (Extended index and little fingers)")
print("- Thumbs Up (One extended thumb)")

print("\n## How to Play")
print("1. The game will display a prompt asking you to make a specific gesture.")

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define a mapping of hand gestures to text
gesture_mapping = {
    2: "Peace Sign",
    3: "Rock and Roll Sign",
    1: "Thumbs Up"
}

# Function to count the number of extended fingers
def count_extended_fingers(hand_landmarks):
    extended_fingers = 0
    finger_tip_ids = [4, 8, 12, 16, 20]
    finger_pip_ids = [5, 9, 13, 17, 0]  # 0 is the wrist landmark

    for tip_id, pip_id in zip(finger_tip_ids, finger_pip_ids):
        tip_x, tip_y = hand_landmarks.landmark[tip_id].x, hand_landmarks.landmark[tip_id].y
        pip_x, pip_y = hand_landmarks.landmark[pip_id].x, hand_landmarks.landmark[pip_id].y

        if pip_y < tip_y:
            extended_fingers += 1

    return extended_fingers

# Continuously monitor hand gestures and convert them to text
cap = cv2.VideoCapture(0)
text_buffer = ""
target_gesture = random.choice(list(gesture_mapping.values()))
previous_gesture = None
success_duration = 0
prompt_text = f"Please make the '{target_gesture}' gesture."
points = 0
end_message_displayed = False

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Analyze hand landmarks to recognize gestures
            num_extended_fingers = count_extended_fingers(hand_landmarks)

            # Check for known gestures in the gesture_mapping
            if num_extended_fingers in gesture_mapping:
                recognized_gesture = gesture_mapping[num_extended_fingers]
                text_buffer = recognized_gesture
                if recognized_gesture == target_gesture:
                    success_duration += 1
                    if success_duration >= 20:  # 1 second at 20 FPS
                        points += 1
                        previous_gesture = target_gesture
                        while True:
                            target_gesture = random.choice(list(gesture_mapping.values()))
                            if target_gesture != previous_gesture:
                                break
                        prompt_text = f"Please make the '{target_gesture}' gesture."
                        success_duration = 0
                        if points == 10 and not end_message_displayed:
                            end_message_displayed = True
                            print("You completed 10 points! Displaying end message.")
                else:
                    success_duration = 0

    if end_message_displayed:
        cv2.putText(image, "Good job! You did it!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Display the image, recognized text, prompt, and points
        cv2.putText(image, text_buffer, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, prompt_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, f"Points: {points}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Gesture Recognition', image)

    # Clear the text buffer after a short delay
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        text_buffer = ""
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
