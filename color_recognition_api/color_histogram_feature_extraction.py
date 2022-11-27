#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 31st December 2017 - new year eve :)
# ----------------------------------------------

from PIL import Image
import os
import cv2
import numpy as np
from color_recognition_api import knn_classifier as knn_classifier
def removeGlare(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
    try:
        result = cv2.inpaint(img, mask, 21, cv2.INPAINT_TELEA)
        return result
    except:
        return img


def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def color_histogram_of_test_image(test_src_image):

    # load the image
    image = test_src_image
    width, height = image.shape[1], image.shape[0]
    crop_size = 10
    # if(width>height): 
    #     image = center_crop(image, (width,crop_size))
    # if(height>width): 
    #     image = center_crop(image, (crop_size,height))
    if(width>height):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        width, height = image.shape[1], image.shape[0]
    image = removeGlare(image)
    image = center_crop(image, (crop_size,height))


    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue
            # print(feature_data)

    with open('test.data', 'w') as myfile:
        myfile.write(feature_data)


def color_histogram_of_training_image(img_name):

    # detect image color by using image file name to label training data
    if 'red' in img_name:
        data_source = 'red'
    elif 'yellow' in img_name:
        data_source = 'yellow'
    elif 'green' in img_name:
        data_source = 'green'
    elif 'orange' in img_name:
        data_source = 'orange'
    elif 'brown' in img_name:
        data_source = 'brown'
    elif 'black' in img_name:
        data_source = 'black'
    elif 'blue' in img_name:
        data_source = 'blue'
    elif 'violet' in img_name:
        data_source = 'violet'
    elif 'gold' in img_name:
        data_source = 'gold'
    elif 'grey' in img_name:
        data_source = 'grey'
    elif 'white' in img_name:
        data_source = 'white'


    # load the image
    image = cv2.imread(img_name)

    width, height = image.shape[1], image.shape[0]
    #crop_size = 10
    # if(width>height): 
    #     image = center_crop(image, (width,crop_size))
    # if(height>width): 
    #     image = center_crop(image, (crop_size,height))
    if(width>height):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        width, height = image.shape[1], image.shape[0]
    image = removeGlare(image)

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    with open('training.data', 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')


def training():

    # red color training images
    for f in os.listdir('./training_dataset/red'):
        color_histogram_of_training_image('./training_dataset/red/' + f)

    # yellow color training images
    for f in os.listdir('./training_dataset/yellow'):
        color_histogram_of_training_image('./training_dataset/yellow/' + f)

    # green color training images
    for f in os.listdir('./training_dataset/green'):
        color_histogram_of_training_image('./training_dataset/green/' + f)

    # orange color training images
    for f in os.listdir('./training_dataset/orange'):
        color_histogram_of_training_image('./training_dataset/orange/' + f)

    # brown color training images
    for f in os.listdir('./training_dataset/brown'):
        color_histogram_of_training_image('./training_dataset/brown/' + f)

    # black color training images
    for f in os.listdir('./training_dataset/black'):
        color_histogram_of_training_image('./training_dataset/black/' + f)

    # blue color training images
    for f in os.listdir('./training_dataset/blue'):
        color_histogram_of_training_image('./training_dataset/blue/' + f)

    # gold color training images
    for f in os.listdir('./training_dataset/gold'):
        color_histogram_of_training_image('./training_dataset/gold/' + f)
        
#    for f in os.listdir('./training_dataset/grey'):
  #      color_histogram_of_training_image('./training_dataset/grey/' + f)	

    for f in os.listdir('./training_dataset/white'):
        color_histogram_of_training_image('./training_dataset/white/' + f)

    for f in os.listdir('./training_dataset/violet'):
        color_histogram_of_training_image('./training_dataset/violet/' + f)	