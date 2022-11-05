from periphery import GPIO

gpio = GPIO("/dev/gpiochip2", 13, "out")

try:
  while True:
    gpio.write(True)
finally:
  gpio.write(False)
  gpio.close()
