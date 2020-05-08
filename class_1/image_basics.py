## This is course material for Introduction to Modern Artificial Intelligence
## Class 1 Example code: image_basics.py
## Author: Allen Y. Yang,  Intelligent Racing Inc.
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

# Step 1: import Python modules. If missing modules, use: conda install <module>
import matplotlib.pyplot as plt
import numpy as np
import os

# Step 2: Read an image
path = os.path.dirname(os.path.abspath(__file__))
filename = path + '/lena512color.tif'
image = plt.imread(filename)
plt.figure(0)
plt.imshow(image)
plt.show()

# Step 3: Show three channels of an RGB image

# make extra copy of image so as to not change the original image values
imageR = np.copy(image)
imageG = np.copy(image)
imageB = np.copy(image)
h, w, channel = image.shape

# set imageR to only nonzero red channel
zeroImage = np.zeros([h,w])
imageR[:,:,1] = zeroImage
imageR[:,:,2] = zeroImage
imageG[:,:,0] = zeroImage
imageG[:,:,2] = zeroImage
imageB[:,:,0] = zeroImage
imageB[:,:,1] = zeroImage

plt.figure(1)
plt.subplot(131)
plt.imshow(imageR)
plt.subplot(132)
plt.imshow(imageG)
plt.subplot(133)
plt.imshow(imageB)
plt.show()

# Step 4. Blur filter
import cv2 as cv

kernel = np.ones([7,7])/49.0
blur_image = cv.filter2D(image, -1, kernel)

plt.figure(2)
plt.imshow(blur_image)
plt.show()