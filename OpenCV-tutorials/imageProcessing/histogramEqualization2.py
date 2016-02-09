'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the image in gray scale
img = cv2.imread('../images/tsukuba_l.png', 0)

# Equalize the image
equalizedImg = cv2.equalizeHist(img)

# Use instead Contrast Limited Adaptive Histogram Equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
claheImg = clahe.apply(img)

# Display the results
plt.subplot(131)
plt.title('original')
plt.imshow(img, 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(132)
plt.title('equalized')
plt.imshow(equalizedImg, 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(133)
plt.title('CLAHE')
plt.imshow(claheImg, 'gray')
plt.xticks([])
plt.yticks([])

plt.show()
