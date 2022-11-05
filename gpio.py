from periphery import GPIO

led = GPIO("/dev/gpiochip2", 13, "out")  # pin 37

try:
  while True:
    led.write(False)
finally:
  led.write(False)
  led.close()
