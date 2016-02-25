'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the image in gray scale
img = cv2.imread('../data/home.jpg', 0)

# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
maskedImg = cv2.bitwise_and(img, img, mask=mask)

# Calculate the histogram with mask and without the mask
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
maskedHist = cv2.calcHist([img], [0], mask, [256], [0, 256])

# Show the images and the histograms
plt.subplot(221)
plt.imshow(img, 'gray')

plt.subplot(222)
plt.imshow(mask, 'gray')

plt.subplot(223)
plt.imshow(maskedImg, 'gray')

plt.subplot(224)
plt.plot(hist)
plt.plot(maskedHist)
plt.xlim([0, 256])

plt.show()
