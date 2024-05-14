# Import Raspberry Pi GPIO library
import RPi.GPIO as GPIO

led_map = {0: 7, 1: 11, 2: 13, 3: 15, 4: 12, 5: 16, 6: 18, 7: 22}


def read_html_file(file_path):
    try:
        with open(file_path, "r") as file:
            html_content = file.read()
        return html_content
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None


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
