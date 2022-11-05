from periphery import GPIO

led = GPIO("/dev/gpiochip2", 13, "out")  # pin 37

while True:
    led.write(True)