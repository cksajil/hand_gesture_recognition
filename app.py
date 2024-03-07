import cv2
import time
import streamlit as st
import mediapipe as mp
from os.path import join
from utils import read_markdown_file
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


def main_page(frame_holder, class_holder, html_holder, raw_frame, gesture_detected):
    if gesture_detected == "Pointing_Up":
        # frame_holder.image(raw_frame, channels="RGB")
        markdown_path = join("static", "index.md")
        html_holder.markdown(read_markdown_file(markdown_path))
        class_holder.write(gesture_detected)
    elif gesture_detected == "Open_Palm":
        # frame_holder.image(raw_frame, channels="RGB")
        markdown_path = join("static", "page_1.md")
        html_holder.markdown(read_markdown_file(markdown_path))
        class_holder.write(gesture_detected)
    elif gesture_detected == "Closed_Fist":
        # frame_holder.image(raw_frame, channels="RGB")
        markdown_path = join("static", "page_2.md")
        html_holder.markdown(read_markdown_file(markdown_path))
        class_holder.write(gesture_detected)
    elif gesture_detected == "Thumb_Up":
        # frame_holder.image(raw_frame, channels="RGB")
        markdown_path = join("static", "page_3.md")
        html_holder.markdown(read_markdown_file(markdown_path))
        class_holder.write(gesture_detected)
    elif gesture_detected == "Thumb_Down":
        # frame_holder.image(raw_frame, channels="RGB")
        markdown_path = join("static", "page_4.md")
        html_holder.markdown(read_markdown_file(markdown_path))
        class_holder.write(gesture_detected)
    elif gesture_detected == "Victory":
        # frame_holder.image(raw_frame, channels="RGB")
        html_holder.markdown("# Page 6 ❄️")
        class_holder.write(gesture_detected)


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
    # st.title("Gesture Controlled Infographics")
    frame_holder = st.empty()
    html_holder = st.empty()
    class_holder = st.empty()
    time.sleep(2)
    while cap.isOpened() and not stop_button_pressed:
        success, raw_frame = cap.read()
        if not success:
            st.write("Video Capture Ended")
            break
        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        gesture_detected = predict_frame(raw_frame)
        main_page(frame_holder, class_holder, html_holder, raw_frame, gesture_detected)
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
