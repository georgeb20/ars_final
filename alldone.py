import argparse
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

import matplotlib as plt
from multiprocessing import cpu_count



def main():
    default_model_dir = '.'
    default_model = 'model.tflite'
    default_labels = 'band-labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 1)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    args = parser.parse_args()

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


    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(args.camera_idx)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)

        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        band_crop = cv2_im.crop(box=bbox)
        opencvImage = np.array(band_crop)
        color_histogram_feature_extraction.color_histogram_of_test_image(opencvImage)
        prediction = knn_classifier.main('training.data', 'test.data')

        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, prediction, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    main()
