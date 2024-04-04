import cv2
import streamlit as st
import mediapipe as mp
from os.path import join
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

pages = [
    "all.png",
    "pc.jpeg",
    "monitor.webp",
    "keyboard.webp",
    "mouse.jpg",
    "harddisk.jpeg",
]


@st.cache_data
def load_image(file):
    return cv2.imread(file)


def main_page(
    frame_holder, class_holder, html_holder, gesture_detected, idx
):
    if gesture_detected in ["Thumb_Up", "Thumb_Down"]:
        if gesture_detected == "Thumb_Up":
            idx += 1
        else:
            idx -= 1
    else:
        pass
    idx = idx % 6
    current_page = pages[idx]
    html_holder.image(load_image(join("static", current_page)), channels="BGR")
    class_holder.write(gesture_detected)
    return idx


def predict_frame(raw_frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_frame)
    recognition_result = recognizer.recognize(mp_image)
    top_gesture = recognition_result.gestures
    gesture_detected = "None"
    if top_gesture:
        gesture_detected = top_gesture[0][0].category_name
    return gesture_detected


def main():
    cap = cv2.VideoCapture(-1)
    width = 16
    height = 16
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    stop_button_pressed = st.button("Stop")
    frame_holder = st.empty()
    html_holder = st.empty()
    class_holder = st.empty()
    idx = 0
    while cap.isOpened() and not stop_button_pressed:
        success, raw_frame = cap.read()
        if not success:
            st.write("Video Capture Ended")
            break
        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        gesture_detected = predict_frame(raw_frame)
        idx = main_page(
            frame_holder, class_holder, html_holder, gesture_detected, idx
        )
        if stop_button_pressed:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
