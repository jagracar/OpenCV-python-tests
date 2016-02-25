'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_histograms/py_2d_histogram/py_2d_histogram.html
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('../data/home.jpg')

# Convert the image to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Calculate the histogram with mask and without the mask
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# Show the images and the histograms
plt.subplot(221)
plt.imshow(img[:, :, ::-1])

plt.subplot(222)
plt.imshow(hist, interpolation='nearest')

plt.show()
