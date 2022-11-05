from periphery import GPIO

led = GPIO("/dev/gpiochip0", 6, "out")  # pin 37

try:
  while True:
    led.write(True)
finally:
  led.write(False)
  led.close()
  button.close()