#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 8th July 2018 - before Google inside look 2018 :)
# -------------------------------------------------------------------------

import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path
import sys
import matplotlib.pyplot as plt
# read the test image
try:
    source_image = cv2.imread(sys.argv[1])
except:
    source_image = cv2.imread('1.png')
prediction = 'n.a.'

# checking whether the training data is ready
PATH = './training.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print ('training data is ready, classifier is loading...')
else:
    print ('training data is being created...')
    open('training.data', 'w')
    color_histogram_feature_extraction.training()
    print ('training data is ready, classifier is loading...')



accuracy=0
count=0
correct=0
for color in os.listdir(os.path.join(os.getcwd(),'test_dataset')):
    path = os.path.join(os.path.join(os.getcwd(),'test_dataset'),color)
    per_color_count=0
    per_color_correct=0
    for file in os.listdir(path):
        #print(color)
        # print(file)
        # get the prediction
        path_img = os.path.join(path,file)
        source_image = cv2.imread(path_img)

        color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
        prediction = knn_classifier.main('training.data', 'test.data')
        # print('Detected color is:', prediction)
        if(prediction==color):
            per_color_correct=per_color_correct+1
            correct=correct+1
        per_color_count=per_color_count+1
        count=count+1
    print(color)
    if(per_color_count !=0):
        print(100*per_color_correct/per_color_count)
print(correct)
print(count)
print(100*correct/count)

print("---------------------------")

print(accuracy)