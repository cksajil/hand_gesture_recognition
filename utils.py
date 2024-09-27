import RPi.GPIO as GPIO

forward_switch_pin = 29
backward_switch_pin = 31
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
    # pass
    GPIO.setwarnings(False)  # Ignore warning for now
    GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
    for key in led_map:
        GPIO.setup(led_map[key], GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(forward_switch_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(backward_switch_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)


def gpio_clear():
    for key in led_map:
        # print("Pin number {} is OFF".format(led_map[key]))
        GPIO.output(led_map[key], GPIO.LOW)


def gpio_action(pin):
    for key in led_map:
        if key == pin:
            # print("Pin number {} is ON".format(led_map[key]))
            GPIO.output(led_map[key], GPIO.HIGH)
        else:
            # print("Pin number {} is OFF".format(led_map[key]))
            GPIO.output(led_map[key], GPIO.LOW)


def cleanup_gpio():
    GPIO.cleanup()


def monitor_inputs(lock):
    global forward_status, backward_status
    while True:
        forward_status = GPIO.input(forward_switch_pin) == GPIO.LOW
        backward_status = GPIO.input(backward_switch_pin) == GPIO.LOW
        # command = (
        #     input("Enter command (f for forward, b for backward, q to quit): ")
        #     .strip()
        #     .lower()
        # )
        # with lock:
        #     if command == "f":
        #         forward_status, backward_status = 1, 0
        #     elif command == "b":
        #         forward_status, backward_status = 0, 1
        #     elif command == "q":
        #         print("Exiting input monitoring.")
        #         break
        #     else:
        #         forward_status, backward_status = 0, 0
