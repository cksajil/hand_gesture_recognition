import os
import cv2
import time
import torch
import logging
import socket
import numpy as np
from PIL import Image
from os.path import join
from threading import Thread
from collections import OrderedDict
from flask_socketio import SocketIO, emit
from flask import Flask, render_template_string, Response
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
from utils import load_config, ConvColumn, setup_gpio, gpio_action, read_html_file

DELAY_COUNT = 10
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
    "home.html",
    "cpu.html",
    "network_card.html",
    "smps.html",
    "motherboard.html",
    "gpu.html",
    "fan.html",
    "storage.html",
    "ram.html",
]

app = Flask(__name__)
log = logging.getLogger("werkzeug")
log.disabled = True
socketio = SocketIO(app)
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
    frame_names = frame_names[offset:num_frames_necessary + offset:step_size]
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

@app.route("/page_content")
def page_content():
    page = current_page["page"]
    page_html = read_html_file(join("static", page))
    return render_template_string(page_html)

def process_video_stream(model, device, transform):
    cap = cv2.VideoCapture(-1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    width = 176
    height = 100
    idx = 0
    n = 0
    gesture_buffer = []
    frames = np.empty((0, height, width, 3))
    gesture_label_int = None
    start_time = time.time()
    try:
        while True:
            success, raw_frame = cap.read()
            if not success:
                print("Video Capture Ended")
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
                target = torch.tensor([2])
                data = data.to(device)

                model.eval()
                output = model(data)

                gesture_label_int, gesture_detected = accuracy(
                    output.detach(), target.detach().cpu(), topk=(1,)
                )
                gesture_buffer.append(gesture_label_int)
                gesture_buffer = gesture_buffer[-30:]
                no_action_count = gesture_buffer.count(0)
                print("no action count:", no_action_count)
                n = 0
                frames = np.empty((0, 100, 176, 3))
                if no_action_count > DELAY_COUNT:
                    check_time = time.time()
                    time_delta = check_time - start_time
                    gesture_buffer.clear()
                    print("no action seconds elapsed:", time_delta)
                    if time_delta > 20:
                        print("Elapsed 20 seconds of inactivity")
                        idx = 0
                if gesture_label_int == 1:
                    start_time = time.time()
                    idx += 1
                elif gesture_label_int == 2:
                    start_time = time.time()
                    idx -= 1

                idx = idx % NUM_PAGES
                page = pages[idx]
                current_page["page"] = page

                gpio_action(idx)

                # Emit the page change event to all connected clients
                socketio.emit("page_change", {"page": page})

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

@app.route('/')
def index():
    page = current_page["page"]
    page_html = read_html_file(join("static", page))
    return render_template_string(
        """
        {{ page_html|safe }}
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.min.js"></script>
        <script type="text/javascript">
            var socket = io();
            socket.on('connect', function() {
                console.log('Connected to server');
            });
            socket.on('page_change', function(data) {
                console.log('Page change to: ' + data.page);
                fetch('/page_content')
                    .then(response => response.text())
                    .then(html => {
                        document.body.innerHTML = html;
                    });
            });
        </script>
    """
    )

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not start video capture.")
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@socketio.on("connect")
def handle_connect():
    page = current_page["page"]
    emit("page_change", {"page": page})

if __name__ == "__main__":
    setup_gpio()
    model = load_model("config.json")
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
    print(f"Server is running at http://{ip_address}:5001")

    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
