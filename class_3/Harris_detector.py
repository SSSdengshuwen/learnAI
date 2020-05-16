## This is course material for Introduction to Modern Artificial Intelligence
## Class 3 Example code: Harris_detector.py
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
cv2.waitKey(0)

# Image smoothing using convolution
smoothFilter = np.ones((7,7))/49.0
smoothImage = cv2.filter2D(image, -1, smoothFilter)
cv2.imshow('Smooth', smoothImage)
cv2.waitKey(0)