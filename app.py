import time
import logging
from os.path import join
from threading import Thread
from utils import read_html_file
from flask_socketio import SocketIO, emit
from flask import Flask, render_template_string
from utils import monitor_inputs
from utils import setup_gpio, gpio_action, cleanup_gpio

NUM_PAGES = 8
SWITCHING_DELAY = 5

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
log = logging.getLogger("werkzeug")
log.disabled = True
socketio = SocketIO(app)
current_page = {"page": pages[0]}


@app.route("/page_content")
def page_content():
    page = current_page["page"]
    page_html = read_html_file(join("static", page))
    return render_template_string(page_html)


def process_video_stream():
    setup_gpio()
    idx = 0
    start_time = time.time()

    while True:
        time.sleep(0.5)
        check_time = time.time()
        time_delta = check_time - start_time
        try:
            input_monitor = monitor_inputs()
            forward_status, backward_status = next(input_monitor)
            if backward_status and not forward_status:
                idx = (idx - 1) % NUM_PAGES
                start_time = time.time()
            elif (
                forward_status and not backward_status
            ) or time_delta > SWITCHING_DELAY:
                print("Elapsed {} seconds or pressed forward".format(SWITCHING_DELAY))
                idx = (idx + 1) % NUM_PAGES
                start_time = time.time()
            else:
                continue
            page = pages[idx]
            current_page["page"] = page
            gpio_action(idx)
            socketio.emit("page_change", {"page": page})

        finally:
            cleanup_gpio()


@app.route("/")
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


@socketio.on("connect")
def handle_connect():
    page = current_page["page"]
    emit("page_change", {"page": page})


if __name__ == "__main__":

    video_thread = Thread(target=process_video_stream)
    video_thread.daemon = True
    video_thread.start()

    # Print the IP address
    # hostname = socket.gethostname()
    # ip_address = socket.gethostbyname(hostname)
    # print(f"Server running on {ip_address}:5001")

    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
