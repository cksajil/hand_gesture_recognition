import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from threading import Thread
from collections import OrderedDict
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
from utils import load_config, ConvColumn, setup_gpio, gpio_action, capture_image

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
    "action.gif",
    "cpu.gif",
    "network_card.gif",
    "smps.gif",
    "motherboard.gif",
    "gpu.gif",
    "fan.gif",
    "storage.gif",
    "ram.gif",
]

current_page = {"page": pages[0]}


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


def display_gif(gif_path):
    cap = cv2.VideoCapture(gif_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        cv2.imshow("GIF Display", frame)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def process_video_stream(model, device, transform):
    width = 176
    height = 100
    idx = 0
    n = 0
    frames = np.empty((0, height, width, 3))
    gesture_label_int = None
    start_time = time.time()

    while True:
        raw_frame = capture_image()
        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        raw_frame = cv2.resize(raw_frame, (176, 100))
        frames = np.append(frames, [raw_frame], axis=0)
        n += 1
        if n % 37 == 0:
            imgs = []
            frames = get_frame_names(frames)
            for frame in frames:
                frame = Image.fromarray((frame * 255).astype(np.uint8))
                frame = transform(frame)
                imgs.append(torch.unsqueeze(frame, 0))

            data = torch.cat(imgs)
            data = data.permute(1, 0, 2, 3)
            data = data[None, :, :, :, :]
            target = torch.tensor([2])
            data = data.to(device)

            output = model(data)
            gesture_label_int, gesture_detected = accuracy(
                output.detach(), target.detach().cpu(), topk=(1,)
            )
            n = 0
            frames = np.empty((0, 100, 176, 3))

            if gesture_label_int in [1, 2]:
                print(gesture_label_int)

            if gesture_label_int == 1:
                idx -= 1
                start_time = time.time()
            elif gesture_label_int == 2:
                idx += 1
                start_time = time.time()
            else:
                check_time = time.time()
                time_delta = check_time - start_time
                time_index = int(time_delta) % 20
                if time_index > 18:
                    print("Elapsed 20 seconds of inactivity")
                    idx = 0
                    start_time = time.time()

            idx = idx % NUM_PAGES
            gif_path = os.path.join(
                "static", pages[idx]
            )  # Updated to include "static" folder
            current_page["page"] = gif_path

            gpio_action(idx)
            display_gif(gif_path)


if __name__ == "__main__":
    setup_gpio()
    model = load_model("config.json")
    model.eval()

    device = torch.device("cpu")
    transform = Compose(
        [
            CenterCrop(84),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    process_video_stream(model, device, transform)
