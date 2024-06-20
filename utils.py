import json
import torch
import base64
import torch.nn as nn
from os.path import join

# import RPi.GPIO as GPIO

led_map = {1: 7, 2: 11, 3: 13, 4: 15, 5: 12, 6: 16, 7: 18, 8: 22}


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


def setup_gpio():
    """Function to set mode for GPIO pins"""
    pass
    # GPIO.setwarnings(False)  # Ignore warning for now
    # GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
    # for key in led_map:
    #     GPIO.setup(led_map[key], GPIO.OUT, initial=GPIO.LOW)


def gpio_clear():
    for key in led_map:
        print("Pin number {} is OFF".format(led_map[key]))
        # GPIO.output(led_map[key], GPIO.LOW)


def gpio_action(pin):
    pass
    # for key in led_map:
    #     if key == pin:
    #         print("Pin number {} is ON".format(led_map[key]))
    #         # GPIO.output(led_map[key], GPIO.HIGH)
    #     else:
    #         print("Pin number {} is OFF".format(led_map[key]))
    #         # GPIO.output(led_map[key], GPIO.LOW)


def load_config(config_name, CONFIG_PATH="./config"):
    with open(join(CONFIG_PATH, config_name)) as file:
        config = json.load(file)
    return config
