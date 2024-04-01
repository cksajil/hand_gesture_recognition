import cv2
import time
import threading
import mediapipe as mp
from os.path import join
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, send_from_directory, render_template_string

model_path = join("models", "gesture_recognizer.task")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)
idx = 0
app = Flask(__name__)

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

pages = [
    "all.png",
    "pc.jpeg",
    "monitor.webp",
    "keyboard.webp",
    "mouse.jpg",
    "harddisk.jpeg",
]


def page_selector(gesture_detected, idx):
    if gesture_detected in ["Thumb_Up", "Thumb_Down"]:
        if gesture_detected == "Thumb_Up":
            idx += 1
        else:
            idx -= 1
    else:
        pass
    idx = idx % 6
    return idx


def predict_frame(raw_frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_frame)
    recognition_result = recognizer.recognize(mp_image)
    top_gesture = recognition_result.gestures
    gesture_detected = "None"
    if top_gesture:
        gesture_detected = top_gesture[0][0].category_name
    return gesture_detected


def detect_gesture():
    global idx
    idx = 0
    cap = cv2.VideoCapture(0)
    width = 128
    height = 128
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    while cap.isOpened():
        success, raw_frame = cap.read()
        if not success:
            break
        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        gesture_detected = predict_frame(raw_frame)
        idx = page_selector(gesture_detected, idx)
        print(idx)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


@app.route("/")
def index():
    threading.Thread(target=detect_gesture).start()
    result_value = idx
    if result_value == 0:
        return send_from_directory("static", "war.mp4")
    else:
        image_path = join("static", pages[result_value])
        with open(join("static", "result.html"), "r") as file:
            template = file.read()
        return render_template_string(template, image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)
