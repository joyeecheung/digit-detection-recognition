#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from PIL import Image, ImageDraw


def detect(im, xml):
    digit_cascade = cv2.CascadeClassifier(xml)
    digits = digit_cascade.detectMultiScale(im)
    return digits


def annotate_detection(im, regions, color=128):
    clone = im.copy()
    draw = ImageDraw.Draw(clone)
    for (x, y, w, h) in regions:
        draw.rectangle((x, y, x+w, y+h), outline=color)
    return clone


def crop_detection(im, regions):
    return [im.crop((x, y, x+w, y+h)) for (x, y, w, h) in regions]

if __name__ == '__main__':
    img = cv2.imread('../asset/test/7.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = Image.open('../asset/test/7.jpg')
    digits = detect(gray, '../asset/classifier2/cascade.xml')
    result = annotate_detection(im, digits)
    result.show()
