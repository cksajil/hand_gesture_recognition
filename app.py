import cv2
import streamlit as st
import mediapipe as mp
from os.path import join
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import read_html_file, generate_gif_content
from utils import setup_gpio, gpio_action, gpio_clear


base_options = python.BaseOptions(model_asset_path="./models/gesture_recognizer.task")
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)
DELAY_COUNT = 10
NUM_PAGES = 8
SELECTED_CLASSES = ["Thumb_Up", "Thumb_Down"]
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
    "cpu.html",
    "network_card.html",
    "smps.html",
    "motherboard.html",
    "gpu.html",
    "fan.html",
    "storage.html",
    "ram.html",
]

gifs = [
    "cpu.gif",
    "network_card.gif",
    "smps.gif",
    "motherboard.gif",
    "gpu.gif",
    "fan.gif",
    "storage.gif",
    "ram.gif",
]


def main_page(gif_holder, html_holder, idx):
    gif_content = generate_gif_content(gifs[idx])
    gif_holder.markdown(gif_content, unsafe_allow_html=True)
    current_page = pages[idx]
    page_html = read_html_file(join("static", current_page))
    if page_html:
        html_holder.markdown(page_html, unsafe_allow_html=True)
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
    setup_gpio()
    cap = cv2.VideoCapture(0)
    width = 16
    height = 16
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    stop_button_pressed = st.button("Stop")
    gif_holder = st.empty()
    html_holder = st.empty()
    idx = 0
    gesture_buffer = []
    while not stop_button_pressed:
        while cap.isOpened() and not stop_button_pressed:
            success, raw_frame = cap.read()
            if not success:
                st.write("Video Capture Ended")
                break
            raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            gesture_detected = predict_frame(raw_frame)
            if gesture_detected not in SELECTED_CLASSES:
                continue
            else:
                gesture_buffer.append(gesture_detected)
                gesture_buffer = gesture_buffer[-DELAY_COUNT:]
                up_count = gesture_buffer.count("Thumb_Up")
                down_count = gesture_buffer.count("Thumb_Down")
                if up_count == DELAY_COUNT:
                    idx += 1
                    gesture_buffer.clear()
                elif down_count == DELAY_COUNT:
                    idx -= 1
                    gesture_buffer.clear()
                idx = idx % NUM_PAGES
            # idx = int(input("Enter page number to load:"))
            idx = main_page(gif_holder, html_holder, idx)
            gpio_action(idx)
            if stop_button_pressed:
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
