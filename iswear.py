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

#serial = Serial("/dev/ttymxc2", 9600)
led = GPIO("/dev/gpiochip2", 13, "out")
led.write(True)

jeff = GPIO("/dev/gpiochip4", 13, "in")

def main():
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        detect_resistor(cap,threshhold=1.3)  
        led.write(False) #stop shaking
        print("Shake off")
        while(jeff.read()==False):
            pass
        focus(cap,threshhold=.5,frames=5)
        print("Shake on")
        led.write(True)
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
    string_res = str(resistance)
    first_digit = string_res[0]
    second_digit = string_res[1]
    num_zeros = str(len(string_res[2:]))
    return first_digit,second_digit,num_zeros

def get_resistance(cv2_im, inference_size, objs, labels):
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
        
    resistance = color2res(colors)
    print(resistance)
    #resistance_array = resistance2array(resistance)
    #serial.write(bytes(resistance_array,'utf-8'))


    print(colors)
    return resistance


def color2res(bands):
    colors =  ["black","brown","red","orange","yellow","green","blue","violet","grey","white","gold"]
    values = [0,1,2,3,4,5,6,7,8,9,-1]

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
        # objs = get_objects(interpreter, args.threshold)[:args.top_k]
