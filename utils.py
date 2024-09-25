import os
import time
import json
import serial
import numpy as np
from os.path import join
import serial.tools.list_ports


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
