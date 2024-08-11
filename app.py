import os
import cv2
import time
import torch
import logging
import socket
import numpy as np
from PIL import Image
import torch.nn as nn
from os.path import join
from threading import Thread
from collections import OrderedDict
from webserver import WebServer  # Importing WebServer from the webserver package
from utils import load_config, ConvColumn, setup_gpio, gpio_action, read_html_file
from utils import capture_image

NUM_PAGES = 9
SELECTED_CLASSES = ["Slide Two Fingers Left", "Slide Two Fingers Right"]
CLASSES = {
    0: "No Gesture",
    1: "Slide Two Fingers Left",
    2: "Slide Two Fingers Right",
    3: "Slide Two Fingers Down",
    4: "Slide Two Fingers Up",
    5: "Shaking Hand",
    6: "Stop Sign",
    7: "Pull Two Fingers In",
}
pages = [
    "home.gif",
    "cpu.gif",
    "network_card.gif",
    "smps.gif",
    "motherboard.gif",
    "gpu.gif",
    "fan.gif",
    "storage.gif",
    "ram.gif",
]

log = logging.getLogger("werkzeug")
log.disabled = True
current_page = {"page": pages[0]}


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    probs, pred = output.cpu().topk(maxk, 1, True, True)
    top_pred = pred[0][0]
    top_prob = probs[0][0].item()
    gesture_detected = CLASSES[top_pred.item()]
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    gesture_label_int = top_pred.item()
    return gesture_label_int, gesture_detected, top_prob


def get_frame_names(frames):
    nclips = 1
    is_val = False
    step_size = 2
    clip_size = 18
    frame_names = frames
    num_frames = len(frames)

    if nclips > -1:
        num_frames_necessary = clip_size * nclips * step_size
    else:
        num_frames_necessary = num_frames

    offset = 0
    if num_frames_necessary > num_frames:
        frame_names += [frame_names[-1]] * (num_frames_necessary - num_frames)
    elif num_frames_necessary < num_frames:
        diff = num_frames - num_frames_necessary
        if not is_val:
            offset = np.random.randint(0, diff)
    frame_names = frame_names[offset : num_frames_necessary + offset : step_size]
    return frame_names


def load_model(config_path):
    config = load_config(config_path)
    model = ConvColumn(8)
    if os.path.isfile(config["checkpoint"]):
        checkpoint = torch.load(config["checkpoint"], map_location="cpu")
        new_state_dict = OrderedDict()

        for k, v in checkpoint.items():
            if k == "state_dict":
                for j, val in v.items():
                    name = j[7:]
                    new_state_dict[name] = val
                model.load_state_dict(new_state_dict)
                break
        print("Loaded checkpoint")
    else:
        print("No checkpoint found at '{}'".format(config["checkpoint"]))
    return model


def process_video_stream(model, device, transform):
    width = 176
    height = 100
    idx = 0
    frames = np.empty((0, height, width, 3))
    window_size = 18  # The number of frames to use for each prediction
    overlap = 2  # The number of overlapping frames between consecutive windows
    threshold = 0.95  # Probability threshold for considering a prediction
    consecutive_count = 6  # Number of consecutive predictions needed to change a page
    gesture_count = {key: 0 for key in CLASSES.keys()}  # Count for each gesture
    current_gesture = None
    start_time = time.time()

    while True:
        raw_frame = capture_image()
        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        raw_frame = cv2.resize(raw_frame, (176, 100))
        frames = np.append(frames, [raw_frame], axis=0)

        # Check if we have enough frames for a prediction
        if len(frames) >= window_size:
            # Extract the window of frames to make a prediction
            frame_window = frames[-window_size:]
            imgs = []

            for frame in frame_window:
                frame = Image.fromarray((frame * 255).astype(np.uint8))
                frame = transform(frame)
                imgs.append(torch.unsqueeze(frame, 0))

            data = torch.cat(imgs)
            data = data.permute(1, 0, 2, 3)
            data = data[None, :, :, :, :]
            target = torch.tensor([2])
            data = data.to(device)

            output = model(data)
            gesture_label_int, gesture_detected, prob = accuracy(
                output.detach(), target.detach().cpu(), topk=(1,)
            )

            if prob >= threshold:
                if gesture_count[gesture_label_int] == 0:
                    # Start counting consecutive predictions
                    current_gesture = gesture_label_int
                if gesture_label_int == current_gesture:
                    gesture_count[gesture_label_int] += 1
                else:
                    # Reset the count for the previous gesture
                    gesture_count[current_gesture] = 0
                    current_gesture = gesture_label_int
                    gesture_count[current_gesture] = 1

                if gesture_count[current_gesture] >= consecutive_count:
                    if current_gesture in [
                        1,
                        2,
                    ]:  # Only change pages for specific gestures
                        print(current_gesture)

                        if current_gesture == 1:
                            idx -= 1
                            start_time = time.time()
                        elif current_gesture == 2:
                            idx += 1
                            start_time = time.time()
                        idx = idx % NUM_PAGES
                        page = pages[idx]
                        current_page["page"] = page

                        gpio_action(idx)
                        socketio.emit("page_change", {"page": page})

                    # Reset gesture count after changing page
                    gesture_count[current_gesture] = 0

            # Slide the window by the overlap amount
            frames = frames[overlap:]


def page_content():
    page = current_page["page"]
    return open(join("static", page), "rb").read()


def handle_page_change(page):
    current_page["page"] = page


def start_server():
    server = WebServer(host="0.0.0.0", port=5001)
    server.add_route("/", lambda: page_content(), "GET")
    server.add_event("page_change", lambda data: handle_page_change(data["page"]))
    server.start()


if __name__ == "__main__":
    setup_gpio()
    model = load_model("config.json")
    model.eval()
    model = torch.jit.script(model)

    device = torch.device("cpu")
    transform = Compose(
        [
            CenterCrop(84),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    video_thread = Thread(target=process_video_stream, args=(model, device, transform))
    video_thread.daemon = True
    video_thread.start()

    # Print the IP address
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    start_server()
