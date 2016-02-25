'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_optimization/py_optimization.html
'''

import numpy as np
import cv2

# Load the image
img1 = cv2.imread('../data/messi5.jpg')

# Get the initial tick count
startCount = cv2.getTickCount()

# Perform the calculations
for kernelSize in range(5, 49, 2):
    img1 = cv2.medianBlur(img1, kernelSize)

# Calculate the elapsed time
time = (cv2.getTickCount() - startCount) / cv2.getTickFrequency()
print(time)
