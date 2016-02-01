'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
'''

import numpy as np
import cv2

# Load the image
img = cv2.imread('images/messi5.jpg')

# Increase the size by a factor of 2
zoomedImg = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Equivalent way to do it
height, width = img.shape[:2]
zoomedImg = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)

# Create a translation
M = np.float32([[1, 0, 100], [0, 1, 50]])
translatedImg = cv2.warpAffine(img, M, (width, height))

# Create a rotation and reduce the image size
M = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 0.5)
rotatedImg = cv2.warpAffine(img, M, (width, height))

# Display all the different images
cv2.imshow('zoomed', zoomedImg)
cv2.imshow('translated', translatedImg)
cv2.imshow('rotated', rotatedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
