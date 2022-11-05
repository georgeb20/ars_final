from periphery import GPIO

gpio_p16 = GPIO("/dev/gpiochip2", 9, "out")

try:
  while True:
    gpio_p16.write(True)
finally:
  gpio_p16.write(False)
  gpio_p16.close()
