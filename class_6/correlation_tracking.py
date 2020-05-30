## This is course material for Introduction to Modern Artificial Intelligence
## Class 6 Example code: correlation_detection.py
## Author: Allen Y. Yang,  Intelligent Racing Inc.
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import numpy as np
import cv2
import os

LOST_TRACK = 0
UPDATE_TRACK = 1

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

            # Only process the largest face
            largestArea = 0
            for i in range(len(faces)):
                if faces[i][2]*faces[i][3] > largestArea:
                    largestArea = faces[i][2]*faces[i][3]
                    largestFace = faces[i]

            # Initiate a tracker
            # Possible choices in cv2:
            # TrackerBoosting, TrackerMIL, TrackerKCF, TrackerTLD, TrackerMOSSE, TrackerCSRT
            tracker = cv2.TrackerKCF_create()
            x = max(largestFace[0]-largestFace[2]//2 , 0)
            y = max(largestFace[1]-largestFace[3]//2, 0)
            w = (largestFace[0]-x)*2 + largestFace[2]
            h = (largestFace[1]-y)*2 + largestFace[3]
            status = tracker.init(frame, (x,y, w, h))
            if status:
                trackingStatus = UPDATE_TRACK
    else:
        # Update Tracker
        status, boundingBox = tracker.update(frame)
        if status:
            x, y, w, h = boundingBox
            x = round(x); y = round(y); w = round(w); h = round(h)
            if x+w>=width: w = width - x - 1
            if y+h>=height: h = height - y -1
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        else:
            # Lost Track
            trackingStatus = LOST_TRACK
            print('Lost Track!')

    cv2.imshow('Video',frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

grabber.release()
cv2.destroyAllWindows()
