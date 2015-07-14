#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image, ImageFont
import cv2
import numpy as np

from detect import detect, crop_detection, annotate_detection
from load_labels import get_data
from recognize import cvtrain, sktrain, preprocess
from recognize import annotate_recognition

SAMPLE_SIZE = (28, 28)
SZ = 28
LABEL_FILE = '../MNIST/train-labels.idx1-ubyte'
IMAGE_FILE = '../MNIST/train-images.idx3-ubyte'
CASCADE_FILE = '../asset/classifier/cascade.xml'
TEST_FILE = '../asset/test/8.jpg'

RESULT_FILE_CV = '../asset/results/cv.jpg'
RESULT_FILE_SK = '../asset/results/sk.jpg'
FONT_FILE = 'arial.ttf'
FONT_SIZE = 30
TEST_FONT = '5'
TRAIN_SIZE = 10000

bin_n = 16  # Number of bins
svm_params = dict(kernel_type=cv2.SVM_LINEAR,
                  svm_type=cv2.SVM_C_SVC,
                  C=2.67, gamma=5.383)

affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR


def main():
    img = cv2.imread(TEST_FILE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = Image.open(TEST_FILE)
    digits = detect(gray, CASCADE_FILE)
    results = crop_detection(im.copy(), digits)
    test = [np.float32(i.resize(SAMPLE_SIZE)).ravel() for i in results]

    images, labels, num, rows, cols = get_data(LABEL_FILE,
                                               IMAGE_FILE)

    font = ImageFont.truetype(FONT_FILE, FONT_SIZE)
    detected = annotate_detection(im.copy(), digits)

    print 'Training OpenCV SVM...'
    svc1 = cvtrain(images[:TRAIN_SIZE], labels[:TRAIN_SIZE], num, rows, cols)
    traindata = preprocess(test, rows, cols).reshape(-1, bin_n * 4)
    yhat1 = svc1.predict_all(traindata)
    yhat1 = yhat1.astype(np.uint8).ravel()

    recognized = annotate_recognition(detected, digits, yhat1, font)
    recognized.show()
    recognized.save(RESULT_FILE_CV)

    print 'Training sklearn SVM...'
    svc2 = sktrain(images[:TRAIN_SIZE], labels[:TRAIN_SIZE])
    yhat2 = svc2.predict(test)

    recognized = annotate_recognition(detected, digits, yhat2, font)
    recognized.show()
    recognized.save(RESULT_FILE_SK)


if __name__ == '__main__':
    main()
