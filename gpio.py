from periphery import GPIO
import time
solenoid = GPIO("/dev/gpiochip2", 13, "out")  # pin 37

try:
  while True:
    solenoid.write(True)
    time.sleep(1)
    solenoid.write(False)
    time.sleep(1)
finally:
  solenoid.write(False)
  solenoid.close()