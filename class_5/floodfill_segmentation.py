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
outputFilename = path + '/rgb_floodfill.avi'

grabber = cv2.VideoCapture(filename)
fps = int(grabber.get(cv2.CAP_PROP_FPS))
width = int(grabber.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(grabber.get(cv2.CAP_PROP_FRAME_HEIGHT))
output = cv2.VideoWriter(outputFilename, cv2.VideoWriter_fourcc('M', 'J', 'P','G'),\
     fps, (width, height), True)

while(grabber.isOpened()):
    ret, frame = grabber.read()

    seed = (width//2, height-100)
    # Flood the frame to segment only the road
    cv2.floodFill(frame, None, seedPoint = seed, newVal = (255, 0, 0), loDiff=(3,3,3), upDiff = (3,3,3))
    
    # Augment the illustration of the seed point
    cv2.circle(frame, seed, 2, (0, 0, 255), cv2.FILLED, cv2.LINE_AA)

    output.write(frame)

    cv2.imshow('Road', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

grabber.release()
output.release()
cv2.destroyAllWindows()