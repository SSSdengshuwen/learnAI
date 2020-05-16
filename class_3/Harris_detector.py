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
filename = path + '/test_image.jpg'
image = cv2.imread(filename, 0)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayImage = grayImage.astype(np.float32)

# Apply Harris corner detection
cornerImage = cv2.cornerHarris(grayImage,2,3,0.04)
image[cornerImage>0.01*cornerImage.max()] = [0,0,255]
cv2.imshow('Harris corner', image)
cv2.waitKey(0)
