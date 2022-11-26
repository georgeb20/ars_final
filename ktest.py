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
# read the test image


# checking whether the training data is ready
PATH = './training.data'


print ('training data is being created...')
open('training.data', 'w')
color_histogram_feature_extraction.training()
print ('training data is ready, classifier is loading...')

count=0
correct=0
for color in os.listdir(os.path.join(os.getcwd(),'test_dataset')):
    path = os.path.join(os.path.join(os.getcwd(),'test_dataset'),color)
    per_color_count=0
    per_color_correct=0
    for file in os.listdir(path):
        path_img = os.path.join(path,file)
        source_image = cv2.imread(path_img)
        color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
        prediction = knn_classifier.main('training.data', 'test.data')
        if(prediction==color):
            per_color_correct=per_color_correct+1
            correct=correct+1
        per_color_count=per_color_count+1
        count=count+1
    if(per_color_count !=0):
        print(color, 100*per_color_correct/per_color_count,"%")

print("---------------------------")

print("Overall Accuracy",100*correct/count,"%")
