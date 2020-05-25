## This is course material for Introduction to Modern Artificial Intelligence
## Class 5 Example code: horizon_detection.py
## Author: Allen Y. Yang,  Intelligent Racing Inc.
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import cv2
import numpy as np
import os

def extract_edge(image):
    # detect line features by taking derivatives in (x,y) plane
    image = image.astype(np.float32)
    f_x = np.array([[1, 0, -1],[2,0,-2], [1, 0, -1]])
    f_y = np.array([[1, 2, 1],[0,0,0], [-1, -2, -1]])

    smoothFilter = np.ones((5,5))/25.0
    smoothImage = cv2.filter2D(image, -1, smoothFilter)
    Ix = cv2.filter2D(smoothImage, -1, f_x)
    Iy = cv2.filter2D(smoothImage, -1, f_y)
    GImage = np.sqrt(Ix**2 + Iy**2)
    GImage = GImage.astype(np.float32)
    outImage = cv2.normalize(GImage, None, 0, 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
    _,filteredImage = cv2.threshold(outImage, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    dilatedImage = cv2.dilate(filteredImage, kernel, iterations = 3)
    erodedImage = cv2.erode(dilatedImage, kernel, iterations = 4)
    return erodedImage

def region_of_interest(image):
    height,width = image.shape
    ROI = np.array([(0, height), (width, height), (width, height-400), (0, height-400)])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [ROI], 255)
    return mask

def display_lines(image, lines):
    if lines is not None:
        for line in lines:
            # print(line)
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return image

# load file
path = os.path.dirname(os.path.abspath(__file__))
filename = path + '/test_image.jpg'
image = cv2.imread(filename)

if  image is None:
    print('Error: image cannot be read. Quit!')
    exit()

# Create gray image and denoise
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edge_image = extract_edge(gray_image)
cv2.imshow('Edge Image', edge_image)
cv2.waitKey(0)

# Create a mask to identify Region of Interest
mask_image = region_of_interest(edge_image)
cv2.imshow('Mask', mask_image)
masked_edge_image = cv2.bitwise_and(edge_image,mask_image)
cv2.imshow('Masked Image', masked_edge_image)
cv2.waitKey(0)

lines = cv2.HoughLinesP(masked_edge_image, 1, np.pi/180, 100, \
    minLineLength = 40, maxLineGap = 5)
lines_image = display_lines(image, lines)
cv2.imshow('Lines', lines_image)
cv2.waitKey(0)
