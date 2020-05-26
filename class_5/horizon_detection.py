## This is course material for Introduction to Modern Artificial Intelligence
## Class 5 Example code: horizon_detection.py
## Author: Allen Y. Yang,  Intelligent Racing Inc.
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import cv2
import numpy as np
import os

def closing(image, dilateIter=1, erodeIter=1):
    # detect line features by taking derivatives in (x,y) plane
    kernel = np.ones((3,3), np.uint8)
    dilatedImage = cv2.dilate(image, kernel, iterations = dilateIter)
    erodedImage = cv2.erode(dilatedImage, kernel, iterations = erodeIter)
    return erodedImage

def region_of_interest(image):
    height,width = image.shape
    ROI = np.array([(0, height-100), (width, height-100), (width-300, height-300), (300, height-300)])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [ROI], 255)
    return mask

def display_lines(image, lines):
    if lines is not None:
        for line in lines:
            # print(line)
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return image

# load file
path = os.path.dirname(os.path.abspath(__file__))
filename = path + '/highway_video.mp4'

grabber = cv2.VideoCapture(filename)
mask_image = None
while(grabber.isOpened()):
    ret, frame = grabber.read()

    if ret == False:
        break

    # Create gray image and denoise
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mask_image is None:
        mask_image = region_of_interest(gray_image)

    #edge_image = extract_edge(gray_image)
    edge_image = cv2.Canny(gray_image, 100, 200)
    ROI_image = cv2.bitwise_and(edge_image,mask_image)
    ROI_image = closing(ROI_image, 3,4)
    cv2.imshow('Edge Image', ROI_image)

    height, width = ROI_image.shape
    lines = cv2.HoughLinesP(ROI_image, 1, np.pi/180, 100, \
        minLineLength = width/10, maxLineGap = width/5)
    lines_image = display_lines(frame, lines)
    cv2.imshow('Lines', lines_image)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

grabber.release()
cv2.destroyAllWindows()
