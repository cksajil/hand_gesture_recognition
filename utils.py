# Import Raspberry Pi GPIO library
import RPi.GPIO as GPIO

led_map = {1: 7, 2: 11, 3: 13, 4: 15, 5: 12, 6: 16, 7: 18, 8: 22}


def read_markdown_file(markdown_path):
    with open(markdown_path, "r") as file:
        markdown_content = file.read()
    return markdown_content


def setup_gpio():
    GPIO.setwarnings(False)  # Ignore warning for now
    GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
    for key in led_map:
        GPIO.setup(led_map[key], GPIO.OUT, initial=GPIO.LOW)


def gpio_clear():
    for key in led_map:
        print("Pin number {} is OFF".format(led_map[key]))
        GPIO.output(led_map[key], GPIO.LOW)


def gpio_action(pin):
    for key in led_map:
        if key == pin:
            print("Pin number {} is ON".format(led_map[key]))
            GPIO.output(led_map[key], GPIO.HIGH)
        else:
            print("Pin number {} is OFF".format(led_map[key]))
            GPIO.output(led_map[key], GPIO.LOW)
