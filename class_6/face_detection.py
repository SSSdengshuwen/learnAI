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
cap = cv2.VideoCapture(filename) # Take-Home: Change to zero to try for yourself!

while 1: 
    # reads frames from a camera
    return_flag, img = cap.read()  
  
    if not(return_flag): # Test if read image is successful
        break
    
    # resize the image to 640
    height, width, channels = img.shape
    height = height // 2
    width = width // 2
    img = cv2.resize(img,(width, height), interpolation = cv2.INTER_CUBIC)

    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.1, 10) 
  
    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
  
    # Display an image in a window 
    cv2.imshow('img',img) 

    #cv2.imwrite('./gray.png',gray) # Please try this to save an image
  
    # Wait for Esc key to stop 
    k = cv2.waitKey(1)
    if k == ord('q'): 
        break
    
# Take-Home Task: save the image only when pressing 's' key

  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  