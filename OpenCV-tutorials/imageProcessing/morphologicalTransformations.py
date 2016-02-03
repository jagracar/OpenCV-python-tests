'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the image in gray scale
img = cv2.imread('../images/j.png', 0)

# Create the desired kernel
kernelType = 1

if kernelType == 0:
    kernel = np.ones((5, 5), np.uint8)
elif kernelType == 1:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
elif kernelType == 1:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
elif kernelType == 2:
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

# Apply the different transformations
erosion = cv2.erode(img, kernel=kernel, iterations=1)
dilation = cv2.dilate(img, kernel=kernel, iterations=1)
opening = cv2.morphologyEx(img, op=cv2.MORPH_OPEN, kernel=kernel)
closing = cv2.morphologyEx(img, op=cv2.MORPH_CLOSE, kernel=kernel)
gradient = cv2.morphologyEx(img, op=cv2.MORPH_GRADIENT, kernel=kernel)
tophat = cv2.morphologyEx(img, op=cv2.MORPH_TOPHAT, kernel=kernel)
blackhat = cv2.morphologyEx(img, op=cv2.MORPH_BLACKHAT, kernel=kernel)

# Display the results
titles = ['original', 'erosion', 'dilation', 'opening', 'closing', 'gradient', 'tophat', 'blackhat']
images = [img, erosion, dilation, opening, closing, gradient, tophat, blackhat]

for i in range(len(titles)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], cmap='gray', interpolation='bicubic')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
