# https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer
# https://github.com/cansik/yolo-hand-detection
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# from yolo import YOLO


# hand_yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
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
# size_new = 256
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
# HAND_COUNT = 1

while True:
    success, raw_frame = cap.read()
    # width, height, inference_time, results = hand_yolo.inference(raw_frame)
    # results.sort(key=lambda x: x[2])

    # if len(results):
    #     for detection in results[:HAND_COUNT]:
    #         id, name, confidence, x, y, w, h = detection
    #         cx = x + (w / 2)
    #         cy = y + (h / 2)

    #         # draw a bounding box rectangle and label on the image
    #         color = (0, 255, 255)
    #         cv2.rectangle(raw_frame, (x, y), (x + w, y + h), color, 2)
    #         text = "%s (%s)" % (name, round(confidence, 2))
    #         cv2.putText(
    #             raw_frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
    #         )
    #         if w > h:
    #             larger = w
    #         elif w < h:
    #             larger = h
    #         else:
    #             larger = w
    #         cropped_image = raw_frame[y : y + larger, x : x + larger, :]
    #         resized_image = cv2.resize(
    #             cropped_image, (size_new, size_new), interpolation=cv2.INTER_AREA
    #         )

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_frame)
    recognition_result = recognizer.recognize(mp_image)
    top_gesture = recognition_result.gestures
    if top_gesture:
        hand_landmarks = recognition_result.hand_landmarks
        print(top_gesture[0][0].category_name)

    cv2.imshow("Image", raw_frame)
    cv2.waitKey(1)
