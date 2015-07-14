#!/usr/bin/env python
# -*- coding: utf-8 -*-

from detect import detect, crop_detection, annotate_detection
from PIL import Image, ImageDraw, ImageFont

import cv2
import numpy as np
from load_labels import get_data
from sklearn import svm

SAMPLE_SIZE = (28, 28)
LABEL_FILE = '../MNIST/train-labels.idx1-ubyte'
IMAGE_FILE = '../MNIST/train-images.idx3-ubyte'
CASCADE_FILE = '../asset/classifier2/cascade.xml'
FONT_FILE = 'arial.ttf'
FONT_SIZE = 30
TEST_FONT = '5'


def train(images, labels):
    svc = svm.SVC(kernel='linear')
    svc.fit(images, labels)
    return svc


def get_font_size(font):
    return max(font.getsize(TEST_FONT))

def annotate_recognition(im, regions, labels, font, color=255):
    clone = im.copy()
    draw = ImageDraw.Draw(clone)
    size = get_font_size(font)
    for idx, (x, y, w, h) in enumerate(regions):
        draw.text((x+w, y+h-size), str(labels[idx]), font=font, fill=color)
    return clone

if __name__ == '__main__':
    img = cv2.imread('../asset/test/7.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = Image.open('../asset/test/7.jpg')
    digits = detect(gray, CASCADE_FILE)
    results = crop_detection(im.copy(), digits)

    images, labels, num, rows, cols = get_data(LABEL_FILE,
                                               IMAGE_FILE)
    test = [np.array(i.resize(SAMPLE_SIZE)).ravel() for i in results]

    svc = train(images[:10000], labels[:10000])
    labels = svc.predict(test)
    font = ImageFont.truetype(FONT_FILE, FONT_SIZE)
    detected = annotate_detection(im.copy(), digits)
    recognized = annotate_recognition(detected, digits, labels, font)
    recognized.show()