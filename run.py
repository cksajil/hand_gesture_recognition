# https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="./models/gesture_recognizer.task")
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)
CLASSES = [
    "None",
    "Closed_Fist",
    "Open_Palm",
    "Pointing_Up",
    "Thumb_Down",
    "Thumb_Up",
    "Victory",
    "ILoveYou",
]
cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)
width = 640
height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while True:
    success, raw_frame = cap.read()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_frame)
    recognition_result = recognizer.recognize(mp_image)
    top_gesture = recognition_result.gestures
    if top_gesture:
        hand_landmarks = recognition_result.hand_landmarks
        print(top_gesture[0][0].category_name)
    cv2.imshow("Image", raw_frame)
    cv2.waitKey(1)
