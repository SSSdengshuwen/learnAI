## This is course material for Introduction to Modern Artificial Intelligence
## Class 3 Example code: Sobel_filter.py
## Author: Allen Y. Yang,  Intelligent Racing Inc.
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import cv2
import numpy as np
import os

# Load the example image
path = os.path.dirname(os.path.abspath(__file__))
filename = path + '/lincoln.tif'
image = cv2.imread(filename)
cv2.imshow('Original', image)

# Image smoothing using convolution
smoothFilter = np.ones((7,7))/49.0
smoothImage = cv2.filter2D(image, -1, smoothFilter)
cv2.imshow('Smooth', smoothImage)

# Apply Sobel filter\
f_x = np.array([[1, 0, -1],[2,0,-2], [1, 0, -1]])
GxImage = cv2.filter2D(image, -1, f_x)
outImage = cv2.normalize(GxImage, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
cv2.imshow('Gx', outImage)

f_y = np.array([[1, 2, 1],[0,0,0], [-1, -2, -1]])
GyImage = cv2.filter2D(image, -1, f_y)
outImage = cv2.normalize(GyImage, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
cv2.imshow('Gy', outImage)

GImage = np.sqrt(np.square(GxImage) + np.square(GyImage))
outImage = cv2.normalize(GImage, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
cv2.imshow('G', outImage)
cv2.waitKey(0)