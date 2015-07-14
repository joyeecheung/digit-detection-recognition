#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import ImageDraw

import cv2
import numpy as np
from sklearn import svm

SAMPLE_SIZE = (28, 28)
SZ = 28
TEST_FONT = '5'

bin_n = 16  # Number of bins
svm_params = dict(kernel_type=cv2.SVM_LINEAR,
                  svm_type=cv2.SVM_C_SVC)

affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img


def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n * ang / (2 * np.pi))
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n)
             for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist


def cvtrain(images, labels, num, rows, cols):
    svc = cv2.SVM()
    traindata = preprocess(images, rows, cols)
    responses = np.float32(labels[:, None])
    svc.train(traindata, responses, params=svm_params)
    return svc


def sktrain(images, labels):
    svc = svm.SVC(kernel='linear')
    svc.fit(images, labels)
    return svc


def preprocess(images, rows, cols):
    deskewed = [deskew(im.reshape(rows, cols)) for im in images]
    hogdata = [hog(im) for im in deskewed]
    return np.float32(hogdata).reshape(-1, 64)


def get_font_size(font):
    return max(font.getsize(TEST_FONT))


def annotate_recognition(im, regions, labels, font, color=255):
    clone = im.copy()
    draw = ImageDraw.Draw(clone)
    size = get_font_size(font)
    for idx, (x, y, w, h) in enumerate(regions):
        draw.text(
            (x+w-size, y+h-size), str(labels[idx]), font=font, fill=color)
    return clone
