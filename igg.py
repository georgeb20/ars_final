import cv2
import numpy as np
from time import sleep
from periphery import GPIO

cap = cv2.VideoCapture(1)

led = GPIO("/dev/gpiochip2", 13, "out")
ret,frame = cap.read()  
first_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
first_mean = np.abs(np.mean(first_gray))
last_mean = first_mean
while(True):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = np.abs(np.mean(gray) - last_mean)
    print(result)
    if result > 1:
        print("Motion detected!")
        print("Started recording.")
        led.write(True)
        sleep(2)
        last_mean = first_mean
    else:
        led.write(False)
    last_mean= np.mean(gray)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()