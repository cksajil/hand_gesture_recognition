import os
import time
import json
import serial
import subprocess
import numpy as np
from PIL import Image
import torch.nn as nn
from os.path import join
import serial.tools.list_ports


class ConvColumn(nn.Module):

    def __init__(self, num_classes):
        super(ConvColumn, self).__init__()
        self.conv_layer1 = self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2))
        self.conv_layer2 = self._make_conv_layer(64, 128, (2, 2, 2), (2, 2, 2))
        self.conv_layer3 = self._make_conv_layer(128, 256, (2, 2, 2), (2, 2, 2))
        self.conv_layer4 = self._make_conv_layer(256, 256, (2, 2, 2), (2, 2, 2))

        self.fc5 = nn.Linear(12800, 512)
        self.fc5_act = nn.ELU()
        self.fc6 = nn.Linear(512, num_classes)

    def _make_conv_layer(self, in_c, out_c, pool_size, stride):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ELU(),
            nn.MaxPool3d(pool_size, stride=stride, padding=0),
        )
        return conv_layer

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)

        x = x.view(x.size(0), -1)

        x = self.fc5(x)
        x = self.fc5_act(x)

        x = self.fc6(x)
        return x


def read_html_file(file_path):
    try:
        with open(file_path, "r") as file:
            html_content = file.read()
        return html_content
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None


def load_config(config_name, CONFIG_PATH="./config"):
    with open(join(CONFIG_PATH, config_name)) as file:
        config = json.load(file)
    return config


def capture_image():
    # Define the image filename
    image_filename = "temp.jpg"

    with open(os.devnull, "w") as devnull:
        subprocess.run(
            [
                "fswebcam",
                "--no-banner",
                "-r",
                "176x100",
                "--jpeg",
                "50",
                image_filename,
            ],
            stdout=devnull,
            stderr=devnull,
        )

    # Open the image using PIL and convert to a NumPy array
    with Image.open(image_filename) as img:
        image_array = np.array(img)
    return image_array


def list_ports():
    """List all available ports and their descriptions."""
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        print(
            f"Port: {port.device}, Description: {port.description}, HWID: {port.hwid}"
        )
    return ports


def find_arduino_port():
    """Automatically detect the Arduino COM port."""
    ports = list_ports()
    for port in ports:
        # Check for specific identifiers for your Arduino
        if "Arduino" in port.description or "FT232R USB UART" in port.description:
            return port.device
    # Check for /dev/ttyUSB0 if it's not detected by description
    for port in ports:
        if "/dev/ttyUSB0" in port.device:
            return port.device
    return None


def communicate_with_arduino(port, idx, baud_rate=9600):
    """Send data to Arduino and control LEDs."""
    try:
        # Open the serial port
        ser = serial.Serial(port, baud_rate, timeout=1)
        print(f"Connected to {port}")
        time.sleep(2)  # Wait for Arduino to reset

        # Send commands to turn on/off LEDs
        while True:
            ser.write(str(idx).encode())  # Send command to Arduino
            time.sleep(0.5)  # Wait for the command to be processed
            response = ser.readline().decode().strip()
            print(f"Arduino response: {response}")

        ser.close()
    except serial.SerialException as e:
        print(f"Error: {e}")
