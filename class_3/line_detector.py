## This is course material for Introduction to Modern Artificial Intelligence
## Class 3 Example code: line_detection.py
## Author: Allen Y. Yang,  Intelligent Racing Inc.
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import cv2
import numpy as np
import os

def extract_edge(image):
    # detect line features by taking derivatives in (x,y) plane
    edge_image = cv2.Canny(image, 50, 150)
    return edge_image

def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([(200, height), (1100, height), (550, 250)])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [triangle], 255)
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
masked_edge_image = cv2.bitwise_and(edge_image,mask_image)
cv2.imshow('Masked Image', masked_edge_image)
cv2.waitKey(0)

lines = cv2.HoughLinesP(masked_edge_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
lines_image = display_lines(image, lines)
cv2.imshow('Lines', lines_image)
cv2.waitKey(0)
