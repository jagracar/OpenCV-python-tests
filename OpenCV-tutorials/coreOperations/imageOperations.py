'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html
'''

import numpy as np
import cv2

# Load two images
img1 = cv2.imread('../data/butterfly.jpg')
img2 = cv2.imread('../data/messi5.jpg')

# Calculate the minimum number of rows and columns
minRows = min(img1.shape[0], img2.shape[0])
minCols = min(img1.shape[1], img2.shape[1])

# Slice the images to have the same size
img1 = img1[:minRows, :minCols]
img2 = img2[:minRows, :minCols]

# Blend the two images
dst = cv2.addWeighted(img1, 0.3, img2, 0.7, 0)

# Display the result
cv2.imshow('blend', dst)

# Load two other images
img1 = cv2.imread('../data/messi5.jpg')
img2 = cv2.imread('../data/opencv_logo.jpg')

# Create the logo background mask 
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, bgMask = cv2.threshold(img2gray, thresh=170, maxval=255, type=cv2.THRESH_BINARY)

# Create the foreground mask, using the inverse of the background mask
fgMask = cv2.bitwise_not(bgMask)

# We want to put the logo on the top-left corner, so we create a ROI
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]

# Black-out the ROI area covered by the logo
img1Bg = cv2.bitwise_and(roi, roi, mask=bgMask)

# Take only region of logo from logo image.
img2Fg = cv2.bitwise_and(img2, img2, mask=fgMask)

# Update the main image with the addition of the two
img1[0:rows, 0:cols] = cv2.add(img1Bg, img2Fg)

# Show the final image in a new window
cv2.imshow('combined', img1)

# Wait for user interaction
cv2.waitKey(0)
cv2.destroyAllWindows()
