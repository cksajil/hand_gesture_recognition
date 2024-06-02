import os
import cv2
import torch
import numpy as np
from PIL import Image
import streamlit as st
from os.path import join
from collections import OrderedDict
from utils import load_config, ConvColumn
from utils import read_html_file, generate_gif_content
from utils import setup_gpio, gpio_action
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor

NUM_PAGES = 8
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


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.cpu().topk(maxk, 1, True, True)
    top_pred = pred[0][0]
    gesture_detected = CLASSES[top_pred.item()]
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    gesture_label_int = top_pred.item()
    return gesture_label_int, gesture_detected


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


def main():
    device = torch.device("cpu")
    config = load_config("config.json")
    setup_gpio()
    cap = cv2.VideoCapture(0)
    width = 176
    height = 100
    stop_button_pressed = st.button("Stop")
    gif_holder = st.empty()
    html_holder = st.empty()
    idx = 0
    n = 0
    frames = np.empty((0, height, width, 3))

    transform = Compose(
        [
            CenterCrop(84),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = ConvColumn(8)

    if os.path.isfile(config["checkpoint"]):
        print("loading checkpoint")
        checkpoint = torch.load(config["checkpoint"], map_location="cpu")
        new_state_dict = OrderedDict()

        for k, v in checkpoint.items():
            if k == "state_dict":
                del checkpoint["state_dict"]
                for j, val in v.items():
                    name = j[7:]
                    new_state_dict[name] = val
                checkpoint["state_dict"] = new_state_dict
                break
        model.load_state_dict(checkpoint["state_dict"])
        print("loaded checkpoint")
    else:
        print("no checkpoint found at '{}'".format(config["checkpoint"]))

    while not stop_button_pressed:
        while cap.isOpened() and not stop_button_pressed:
            success, raw_frame = cap.read()
            if not success:
                st.write("Video Capture Ended")
                break
            raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            raw_frame = cv2.resize(raw_frame, (176, 100))
            frames = np.append(frames, [raw_frame], axis=0)
            n += 1
            if n == 37:
                imgs = []
                frames = get_frame_names(frames)
                for frame in frames:
                    frame = Image.fromarray((frame * 255).astype(np.uint8))
                    frame = transform(frame)
                    imgs.append(torch.unsqueeze(frame, 0))

                data = torch.cat(imgs)
                data = data.permute(1, 0, 2, 3)
                data = data[None, :, :, :, :]
                target = [2]
                target = torch.tensor(target)
                data = data.to(device)

                model.eval()
                output = model(data)

                gesture_label_int, gesture_detected = accuracy(
                    output.detach(), target.detach().cpu(), topk=(1, 5)
                )
                n = 0
                frames = frames = np.empty((0, 100, 176, 3))
                if gesture_label_int == 2:
                    idx += 1
                elif gesture_label_int == 1:
                    idx -= 1
                idx = idx % NUM_PAGES
                idx = main_page(gif_holder, html_holder, idx)
                gpio_action(idx)
            if stop_button_pressed:
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
