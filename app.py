import time
from threading import Thread, Lock
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
from utils import read_html_file, setup_gpio, gpio_action, cleanup_gpio, monitor_inputs


led_map = {0: 7, 1: 11, 2: 13, 3: 15, 4: 12, 5: 16, 6: 18, 7: 22}

NUM_PAGES = 8
SWITCHING_DELAY = 10
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

app = Flask(__name__)
socketio = SocketIO(app)
current_page = {"page": pages[0]}
input_lock = Lock()
forward_status = 0
backward_status = 0


@app.route("/page_content")
def page_content():
    page = current_page["page"]
    page_html = read_html_file(f"static/{page}")
    return render_template_string(page_html)


def process_video_stream():
    setup_gpio()
    global forward_status, backward_status

    idx = 0
    start_time = time.time()

    while True:
        time.sleep(0.5)
        check_time = time.time()
        time_delta = check_time - start_time

        with input_lock:
            if backward_status and not forward_status:
                idx = (idx - 1) % NUM_PAGES
                start_time = time.time()
            elif forward_status and not backward_status or time_delta > SWITCHING_DELAY:
                idx = (idx + 1) % NUM_PAGES
                start_time = time.time()

            page = pages[idx]
            current_page["page"] = page
            gpio_action(idx)
            socketio.emit("page_change", {"page": page})


@app.route("/")
def index():
    page = current_page["page"]
    page_html = read_html_file(f"static/{page}")
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


@socketio.on("connect")
def handle_connect():
    page = current_page["page"]
    emit("page_change", {"page": page})


def main():
    # Start the input monitoring thread
    input_thread = Thread(target=monitor_inputs, args=(input_lock,))
    input_thread.daemon = True
    input_thread.start()

    # Start the video stream processing thread
    video_thread = Thread(target=process_video_stream)
    video_thread.daemon = True
    video_thread.start()

    socketio.run(app, host="0.0.0.0", port=5001, debug=True)


if __name__ == "__main__":
    main()
