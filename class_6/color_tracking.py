## This is course material for Introduction to Modern Artificial Intelligence
## Class 6 Example code: face_detection.py
## Author: Allen Y. Yang,  Intelligent Racing Inc.
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import numpy as np
import cv2
import os

LOST_TRACK = 0
INIT_TRACK = 1
UPDATE_TRACK = 2

path = os.path.dirname(os.path.abspath(__file__))
filename = path + '/jump_rope.mp4'
XMLAddress = path+'/haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(XMLAddress) 
grabber = cv2.VideoCapture(filename) # Take-Home: Change to zero to try for yourself!
width = int(grabber.get(cv2.CAP_PROP_FRAME_WIDTH))//2
height = int(grabber.get(cv2.CAP_PROP_FRAME_HEIGHT))//2

trackingStatus = LOST_TRACK
while (grabber.isOpened()): 
    
        # reads frames from a camera
    return_flag, frame = grabber.read()  
  
    if not(return_flag): # Test if read image is successful
        break

     # resize the image to 640
    frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_CUBIC)

    if trackingStatus == LOST_TRACK:
        # The face is not being tracked, detect
        # convert to gray scale of each frames 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  
        # Detects faces of different sizes in the input image 
        faces = face_cascade.detectMultiScale(gray, 1.1, 10)

        if len(faces)>0:
            trackingStatus = INIT_TRACK
            largestArea = 0
            for i in range(len(faces)):
                if faces[i][2]*faces[i][3] > largestArea:
                    largestArea = faces[i][2]*faces[i][3]
                    largestFace = faces[i]

            x,y,w,h = largestFace 
            faceSizeThreshold = w * h

    if trackingStatus == INIT_TRACK or trackingStatus == UPDATE_TRACK:
        # Initialize the tracker
        seed = (x + w//2, y + h//2)
        mask = np.zeros((height+2, width+2), dtype = np.uint8)
        fillValue = 255
        cv2.floodFill(frame, mask, seedPoint = seed, newVal = 255, loDiff=(100,10,5), upDiff = (100,10,5), flags = 8 | cv2.FLOODFILL_MASK_ONLY| (fillValue << 8))
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        cv2.imshow('Mask', mask)
        #_, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x, y, w, h = cv2.boundingRect(mask)
        x = x - 1
        y = y - 1
        if w*h<faceSizeThreshold/2 or w*h>faceSizeThreshold*2:
            # Lost Track
            trackingStatus = LOST_TRACK
            # continue
        else:
            trackingStatus = UPDATE_TRACK
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

    cv2.imshow('Video',frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

grabber.release()
cv2.destroyAllWindows()
