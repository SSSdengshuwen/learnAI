## This is course material for Introduction to Modern Artificial Intelligence
## Class 5 Example code: canny_detector.py
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

def region_of_interest(image, height= None, width = None):
    if image is None:
        if (height is None) or (width is None):
            return None
    else:
        height,width = image.shape

    ROI = np.array([(0, height-100), (width, height-100), (width, height-300), (0, height-300)])
    mask = np.zeros([height, width], dtype = np.uint8)
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
outputFilename = path + '/output_video.avi'

grabber = cv2.VideoCapture(filename)
fps = int(grabber.get(cv2.CAP_PROP_FPS))
width = int(grabber.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(grabber.get(cv2.CAP_PROP_FRAME_HEIGHT))
output = cv2.VideoWriter(outputFilename, cv2.VideoWriter_fourcc('M', 'J', 'P','G'), fps, (width, height), False)
mask_image = region_of_interest(None, height, width)
while(grabber.isOpened()):
    ret, frame = grabber.read()

    if ret == False:
        break

    # Create gray image and denoise
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #edge_image = extract_edge(gray_image)
    edge_image = cv2.Canny(gray_image, 100, 200, L2gradient = True)
    ROI_image = cv2.bitwise_and(edge_image,mask_image)
    #ROI_image = closing(ROI_image, 3,4)
    cv2.imshow('Edge Image', ROI_image)
    output.write(ROI_image)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

grabber.release()
output.release()
cv2.destroyAllWindows()
