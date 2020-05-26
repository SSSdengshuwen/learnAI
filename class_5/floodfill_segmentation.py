## This is course material for Introduction to Modern Artificial Intelligence
## Class 5 Example code: floodfill_segmentation.py
## Author: Allen Y. Yang,  Intelligent Racing Inc.
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import cv2
import numpy as np
import os


# load file
path = os.path.dirname(os.path.abspath(__file__))
filename = path + '/highway_video.mp4'

grabber = cv2.VideoCapture(filename)
fps = int(grabber.get(cv2.CAP_PROP_FPS))
width = int(grabber.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(grabber.get(cv2.CAP_PROP_FRAME_HEIGHT))

while(grabber.isOpened()):
    ret, frame = grabber.read()

    seed = (width//2, height-200)
    # Create gray image and denoise
    cv2.floodFill(frame, None, seedPoint = seed, newVal = (255, 0, 0), loDiff=(5,5,5), upDiff = (5,5,5))

    cv2.imshow('Lines', frame)
    cv2.waitKey(0)

grabber.release()
cv2.destroyAllWindows()