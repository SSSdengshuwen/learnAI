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
Gx = np.array([[1, 0, -1],[2,0,-2], [1, 0, -1]])
GxImage = cv2.filter2D(image, -1, Gx)
cv2.imshow('Gx', GxImage)

Gy = np.array([[1, 2, 1],[0,0,0], [-1, -2, -1]])
GyImage = cv2.filter2D(image, -1, Gy)
cv2.imshow('Gy', GyImage)
cv2.waitKey(0)