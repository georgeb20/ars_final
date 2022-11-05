from periphery import GPIO

gpio = GPIO("/sys/class/gpio/gpio77", 13, "out")

try:
  while True:
    gpio.write(True)
finally:
  gpio.write(False)
  gpio.close()
