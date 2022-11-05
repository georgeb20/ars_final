from periphery import GPIO

button = GPIO("/dev/gpiochip4", 13, "in")  # pin 36

try:
  while True:
    button.write(True)
finally:
  button.write(False)
  button.close()
