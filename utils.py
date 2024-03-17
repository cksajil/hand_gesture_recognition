# Import Raspberry Pi GPIO library
import RPi.GPIO as GPIO


def read_markdown_file(markdown_path):
    with open(markdown_path, "r") as file:
        markdown_content = file.read()
    return markdown_content


def setup_gpio():
    GPIO.setwarnings(False)  # Ignore warning for now
    GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering

    for pin in range(11, 17):
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        # Set pin to be an output pin and set initial value to low (off)


def gpio_action(code):
    pins = list(range(range(11, 17)))
    for pin, value in zip(pins, code):
        GPIO.output(pin, value)
