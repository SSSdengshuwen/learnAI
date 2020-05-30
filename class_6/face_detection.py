## This is course material for Introduction to Modern Artificial Intelligence
## Class 6 Example code: face_detection.py
## Author: Allen Y. Yang,  Intelligent Racing Inc.
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import numpy as np
import cv2
import os

path = os.path.dirname(os.path.abspath(__file__))
filename = path + '/jump_rope.mp4'
XMLAddress = path+'/haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(XMLAddress) 
grabber = cv2.VideoCapture(filename) # Take-Home: Change to zero to try for yourself!

while (grabber.isOpened()): 
    # reads frames from a camera
    return_flag, frame = grabber.read()  
  
    if not(return_flag): # Test if read image is successful
        break
    
    # resize the image to 640
    height, width, channels = frame.shape
    height = height // 2
    width = width // 2
    frame = cv2.resize(frame,(width, height), interpolation = cv2.INTER_CUBIC)

    # convert to gray scale of each frames 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.1, 10) 
  
    if len(faces)>0:
        largestArea = 0
        for i in range(len(faces)):
            if faces[i][2]*faces[i][3] > largestArea:
                largestArea = faces[i][2]*faces[i][3]
                largestFace = faces[i]

        x,y,w,h = largestFace 
        # To draw a rectangle in a face  
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
  
    # Display an image in a window 
    cv2.imshow('Face Video', frame) 
  
    # Wait for Esc key to stop 
    key = cv2.waitKey(1)
    if key == ord('q'): 
        break
      
# Close the window 
grabber.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  