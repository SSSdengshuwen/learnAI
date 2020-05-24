## This is course material for Introduction to Modern Artificial Intelligence
## Class 4 Example code: image_morphing.py
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
if  image is None:
    print('Error: image cannot be read. Quit!')
    exit()

cv2.imshow('Original', image)
image = image.astype(np.float32)

# Apply omnidirecitional Sobel filter
f_x = np.array([[1, 0, -1],[2,0,-2], [1, 0, -1]])
f_y = np.array([[1, 2, 1],[0,0,0], [-1, -2, -1]])

smoothFilter = np.ones((5,5))/25.0
smoothImage = cv2.filter2D(image, -1, smoothFilter)
Ix = cv2.filter2D(smoothImage, -1, f_x)
Iy = cv2.filter2D(smoothImage, -1, f_y)
GImage = np.sqrt(Ix**2 + Iy**2)
GImage = GImage.astype(np.float32)
outImage = cv2.normalize(GImage, None, 0, 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
cv2.imshow('Sobel Edge', outImage)

# Filter only the strong edges
_,filteredImage = cv2.threshold(outImage, 127, 255, cv2.THRESH_BINARY)#上一个语句行的元素，可以除2来做中值，低于者为0，255可以设为1但是打印不便//
cv2.imshow('binary Image', filteredImage)

# Dilate and Erode
kernel = np.ones((5,5), np.uint8)
dilatedImage = cv2.dilate(filteredImage, kernel, iterations = 3)
cv2.imshow('dilated Image', dilatedImage)

erodedImage = cv2.erode(dilatedImage, kernel, iterations = 4)
cv2.imshow('eroded Image', erodedImage)
cv2.waitKey(0)