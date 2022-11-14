import argparse
import cv2
import os

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os.path
import numpy as np

from scipy import ndimage
from periphery import Serial
from periphery import GPIO
from time import sleep




serial = Serial("/dev/ttymxc2", 9600)
led = GPIO("/dev/gpiochip2", 13, "out")
led.write(True)
def main():
    
    default_model_dir = '.'
    default_model = 'band80.tflite'
    default_labels = 'labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=4,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 1)
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='classifier score threshold')
    args = parser.parse_args()

    colors_array = ["black","brown","red","orange","yellow","green","blue","violet","grey","white","gold"]
    values = [0,1,2,3,4,5,6,7,8,9,-1]

    prediction = 'n.a.'
    mean = [None]
    sliding_window = []
    filter_type = 'zone'
    # checking whether the training data is ready

    print ('training data is being created...')
    open('training.data', 'w')
    color_histogram_feature_extraction.training()
    print ('training data is ready, classifier is loading...')


    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(args.camera_idx)

    last_mean = 0
    print('hi')
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = np.abs(np.mean(gray) - last_mean)
        print(result)

        if result > .8:
            print(result)
            print("Motion detected!")
            print("Started recording.")
            led.write(False)
            sleep(3)
            res_mean = []
            last_mean=0
            while(True):
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                result = np.abs(np.mean(gray) - last_mean) 
                if(result<1.3):
                    res_mean.append(result)
                    if(len(res_mean)==5):
                        led.write(True)
                        break
                else:
                    res_mean=[]
                last_mean = np.mean(gray)
            


        else:
            led.write(True)
        last_mean= np.mean(gray)


    led.write(False)
    led.close()
    serial.close()
    cap.release()
    cv2.destroyAllWindows()
    

def resistance2array(resistance):
    string_res = str(resistance)
    first_digit = string_res[0]
    second_digit = string_res[1]
    num_zeros = str(len(string_res[2:]))
    return first_digit,second_digit,num_zeros

def append_objs_to_img(cv2_im, inference_size, objs,colors_array,values):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    colors=[]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)

        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)


        dx = x1-x0
        dy = y1-y0

        band_crop = cv2_im[y0:y0+dy,x0:x0+dx]
        color_histogram_feature_extraction.color_histogram_of_test_image(band_crop)
        prediction = knn_classifier.main('training.data', 'test.data')
        colors.append(prediction)
        
    resistance = color2res(colors,colors_array,values)
    #print(resistance)
    #resistance_array = resistance2array(resistance)
    #serial.write(bytes(resistance_array,'utf-8'))
  #  print(colors)

def is_good_photo(img, height, mean, sliding_window):
    threshold = 4.5
    center = ndimage.measurements.center_of_mass(img)
    detection_zone_avg = (center[0] + center[1]) / 2



    if len(sliding_window) > 30:
        mean[0] = np.mean(sliding_window)
        sliding_window.clear()

    else:
        sliding_window.append(detection_zone_avg)
    # print(detection_zone_avg)
    if mean[0] != None and abs(detection_zone_avg - mean[0]) > threshold:
        return True

    return False
def color2res(bands,colors,values):
    if "unknown" in bands:
        return 0
    colors.reverse()
    flag=0
    if len(bands)==4 or len(bands)==5:
        if(bands[0]=="gold"):
          bands.reverse()
          flag=1  
        if(len(bands)==4):
            resistance =  (values[colors.index(bands[0])]*10 + values[colors.index(bands[1])]) * pow(10,(values[colors.index(bands[2])]))
        else:
            resistance =  (values[colors.index(bands[0])]*100 + values[colors.index(bands[1])]*10+values[colors.index(bands[2])]) * pow(10,(values[colors.index(bands[2])]))

        if flag==1:
          bands.reverse()
        return resistance

if __name__ == '__main__':
    main()
