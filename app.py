import cv2
import time
import streamlit as st
import mediapipe as mp
from os.path import join
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="./models/gesture_recognizer.task")
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)
NUM_PAGES = 5
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

pages = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]


def play_video(frame_holder, html_holder, class_holder):
    video_html = """<video controls width="720" autoplay="true" muted="false" loop="true">
<source 
            src="https://github.com/cksajil/hand_gesture_recognition/raw/video/static/war.mp4" 
            type="video/mp4" />
</video>"""
    frame_holder.markdown(video_html, unsafe_allow_html=True)
    html_holder.write("")
    class_holder.write("")


def main_page(
    frame_holder, class_holder, html_holder, raw_frame, gesture_detected, idx
):
    if gesture_detected in ["Thumb_Up", "Thumb_Down"]:
        if gesture_detected == "Thumb_Up":
            idx += 1
        else:
            idx -= 1
    else:
        pass
    idx = idx % NUM_PAGES
    if idx == 0:
        play_video(frame_holder, html_holder, class_holder)
        time.sleep(5)
    else:
        current_page = pages[idx]
        frame_holder.write("")
        html_holder.image(join("static", current_page))
        class_holder.write(gesture_detected)
        time.sleep(3)
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
    cap = cv2.VideoCapture(0)
    width = 128
    height = 128
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    stop_button_pressed = st.button("Stop")
    frame_holder = st.empty()
    html_holder = st.empty()
    class_holder = st.empty()
    time.sleep(3)
    idx = 0
    while cap.isOpened() and not stop_button_pressed:
        success, raw_frame = cap.read()
        if not success:
            break
        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        gesture_detected = predict_frame(raw_frame)
        idx = main_page(
            frame_holder, class_holder, html_holder, raw_frame, gesture_detected, idx
        )
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
