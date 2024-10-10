import cv2
import mediapipe as mp

# Import the tasks API for gesture recognition
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2

import pyautogui

import custom_gestures

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally and convert the BGR image to RGB.
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)

            # Draw the hand annotations on the image.
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    gesture = custom_gestures.recognize_gesture(hand_landmarks)

                    # Example of pressing keys with pyautogui based on recognized gesture
                    if gesture == "Unknown":
                        pyautogui.keyUp("d")
                        pyautogui.keyUp("a")
                    elif gesture == "left_finger":
                        pyautogui.keyDown("a") #move left
                    elif gesture == "right_finger":
                        pyautogui.keyDown("d") #move left
                    elif gesture == "left_and_jump":
                        pyautogui.keyDown("a") #move left
                        pyautogui.press("space") #jump
                    elif gesture == "right_and_jump":
                        pyautogui.keyDown("d") #move right
                        pyautogui.press("space") #jump
                    else:
                        pyautogui.keyUp("d")
                        pyautogui.keyUp("a")
                        if gesture == "thumb_up":
                            pyautogui.press("space") #jump up
                        elif gesture == "open_palm":
                            pyautogui.press("e") #rotate clockwise
                        elif gesture == "okay":
                            pyautogui.press("q") #rotate counterclockwise
                        elif gesture == "peace":
                            pyautogui.press("s") #enter door

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()