import cv2
from scipy import ndimage
import numpy as np
import scipy
from time import sleep
filter_type = 'zone'

def is_good_photo(img, width, height, mean, sliding_window):
    detection_zone_height = 20
    detection_zone_interval = 5
    threshold = 4.5
    if (filter_type == 'zone'):
        detection_zone_avg = img[height // 2 : (height // 2) + detection_zone_height : detection_zone_interval, 0:-1:3].mean()
    if (filter_type == 'biquad2d'):
        detection_zone_avg = abs(bq.process(img.mean))
    if (filter_type == 'biquad'):
        detection_zone_avg = abs(bq.process(img[height // 2: (height // 2) + detection_zone_height: detection_zone_interval, 0:-1:3].mean()))
    if (filter_type == 'center_of_mass'):
        center = scipy.ndimage.measurements.center_of_mass(img)
        detection_zone_avg = (center[0] + center[1]) / 2

    
    if len(sliding_window) > 30:
        if(mean[0]!=None):
            return
        mean[0] = np.mean(sliding_window)
        print("Done setting up")
    else:
        sliding_window.append(detection_zone_avg)

    if mean[0] != None and abs(detection_zone_avg - mean[0]) > threshold:
        print(abs(detection_zone_avg-mean[0]))

        print("Target Detected Taking Picture")
        return True

    return False


mean = [None]
sliding_window = []
cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret,frame = cap.read()
    height,width,channel = frame.shape
    if(is_good_photo(frame,width,height,mean,sliding_window)):
        sleep(5)
        print("Awake")




