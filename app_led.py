import cv2
import time
import streamlit as st
import mediapipe as mp

# import RPi.GPIO as GPIO
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# from utils import setup_gpio, gpio_action

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
        html_holder.image("./static/all.png", caption="Components of Desktop Computer")
        class_holder.markdown(gesture_detected)
        # gpio_action([GPIO.HIGH, GPIO.LOW, GPIO.LOW, GPIO.LOW, GPIO.LOW])

    elif gesture_detected == "Open_Palm":
        # frame_holder.image(raw_frame, channels="RGB")
        html_holder.image("./static/pc.jpeg", caption="Central Processing Unit (CPU)")
        class_holder.write(gesture_detected)
        # gpio_action([GPIO.LOW, GPIO.HIGH, GPIO.LOW, GPIO.LOW, GPIO.LOW])

    elif gesture_detected == "Closed_Fist":
        # frame_holder.image(raw_frame, channels="RGB")
        html_holder.image("./static/monitor.webp", caption="Monitor")
        class_holder.write(gesture_detected)
        # gpio_action([GPIO.LOW, GPIO.LOW, GPIO.HIGH, GPIO.LOW, GPIO.LOW])

    elif gesture_detected == "Thumb_Up":
        # frame_holder.image(raw_frame, channels="RGB")
        html_holder.image("./static/keyboard.webp", caption="Keyboard")
        class_holder.write(gesture_detected)
        # gpio_action([GPIO.LOW, GPIO.LOW, GPIO.LOW, GPIO.HIGH, GPIO.LOW])

    elif gesture_detected == "Thumb_Down":
        # frame_holder.image(raw_frame, channels="RGB")
        html_holder.image("./static/mouse.jpg", caption="Mouse")
        class_holder.write(gesture_detected)
        # gpio_action([GPIO.LOW, GPIO.LOW, GPIO.LOW, GPIO.LOW, GPIO.HIGH])

    elif gesture_detected == "Victory":
        # frame_holder.image(raw_frame, channels="RGB")
        html_holder.image("./static/harddisk.jpeg", caption="Harddisk")
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
    # setup_gpio()
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
