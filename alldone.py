import argparse
import colorsys
import cv2
import os

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
#
import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import numpy as np
import tflite_runtime.interpreter as tflite 
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
import random
import os
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import cv2

import time
from multiprocessing import cpu_count

from scipy import ndimage
from periphery import Serial
from periphery import GPIO
from time import sleep

from statistics import mode


#from utils import CameraWebsocketHandler
#from utils.BiQuad import BiQuadFilter
#from functools import partial
#from PIL import Image
#from scipy import ndimage
#import edgetpu.classification.engine
#import threading
#import asyncio
#import base64
#import utils
#import cv2
#import argparse
#import sys
#import RPi.GPIO as GPIO

resistors = [100,120,270,470,1000,1500,2200,2700,3900,5600,8200,10000,11000,15000,22000,27000,47000,68000,100000,110000,222000,390000,680000,1000000,4700000,5600000,10000000]


def main():
    
    default_model_dir = '.'
    default_model = 'band80.tflite'
    default_labels = 'labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=5,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 1)
    parser.add_argument('--threshold', type=float, default=0.28,
                        help='classifier score threshold')
    args = parser.parse_args()

    colors_array = ["black","brown","red","orange","yellow","green","blue","violet","grey","white","gold"]
    values = [0,1,2,3,4,5,6,7,8,9,-1]

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
    serial = Serial("/dev/ttymxc2", 9600)

    cap = cv2.VideoCapture(args.camera_idx)
    while cap.isOpened():
       # detect_resistor(cap,threshhold=1.3)
       # attempts=0
       # computed_resistance = []
       # final_resistance = 0
       # while(attempts<1): #5 attempts to find the resistance
        focus(cap,threshhold=.3,frames=7)
        ret, cv2_im = cap.read()
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
           # if(len(objs)>5):
           #     print("Multiple resistors detected!")
           #     computed_resistance = []
          #      break
          #  else:
        cv2_im,resistance = append_objs_to_img(cv2_im, inference_size, objs, labels,colors_array,values)
        # if(resistance in resistors):
        #     computed_resistance.append(resistance)
        #     attempts+=1
        # if(computed_resistance == []):
        #     final_resistance = 0
        # else:
        #     final_resistance = mode(computed_resistance)
       # print("Final resistance is ",final_resistance)
        print("Resistance calculated is ", resistance)
       # resistance_array = resistance2array(final_resistance)
        #serial.write(bytes(resistance_array,'utf-8'))
       # a= input("wait")
        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def detect_resistor(cap,threshhold):
    focus(cap,threshhold=.5,frames=5)
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    last_mean = np.mean(gray)
    print("I'm ready to detect a resistor.")
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = np.abs(np.mean(gray) - last_mean)
        if(result>threshhold):
            print("Resistor detected!")        
            return
        last_mean = np.mean(gray)
def focus(cap,threshhold,frames):
    last_mean=0
    count=0
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = np.abs(np.mean(gray) - last_mean) 
        if(result<threshhold):
            count+=1
            if(count==frames):
                return
        else:
            count=0
        last_mean = np.mean(gray)
        

def resistance2array(resistance):
    if(resistance==0):
        return str(0)
    string_res = str(resistance)
    first_digit = string_res[0]
    second_digit = string_res[1]
    num_zeros = str(len(string_res[2:]))
    return first_digit+second_digit+num_zeros


def append_objs_to_img(cv2_im, inference_size, objs, labels,colors_array,values):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    colors=[]
    s=30
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)

        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)


        dx = x1-x0
        dy = y1-y0

        band_crop = cv2_im[y0:y0+dy,x0:x0+dx]
        color_histogram_feature_extraction.color_histogram_of_test_image(band_crop)
        prediction = knn_classifier.main('training.data', 'test.data',3)
        
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, prediction, (s, 90),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        colors.append(prediction)
        s=s+100
        
    resistance = color2res(colors,colors_array,values)
    #resistance_array = resistance2array(resistance)
    #serial.write(bytes(resistance_array,'utf-8'))


    cv2_im = cv2.putText(cv2_im, str(resistance), (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    print(colors)
    return cv2_im,resistance


def color2res(bands,colors,values):
    if "null" in bands:
        return 0
    colors.reverse()
    flag=0
    if len(bands)==4 or len(bands)==5:
        if(bands[0]=="gold"):
          bands.reverse()
        #  flag=1  
      #  if(len(bands)==4):
        resistance =  (values[colors.index(bands[0])]*10 + values[colors.index(bands[1])]) * pow(10,(values[colors.index(bands[2])]))
       # else:
      #      resistance =  (values[colors.index(bands[0])]*100 + values[colors.index(bands[1])]*10+values[colors.index(bands[2])]) * pow(10,(values[colors.index(bands[2])]))

     #   if flag==1:
      #    bands.reverse()
        return resistance
    else:
        return 0

if __name__ == '__main__':
    main()
        # objs = get_objects(interpreter, args.threshold)[:args.top_k]
