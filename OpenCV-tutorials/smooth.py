'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_filtering/py_filtering.html
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('images/opencv_logo.jpg')

# Create the uniform kernel
kernel = np.ones((5, 5), np.float32) / 25

# Apply the different filters
uniformImg = cv2.filter2D(img, -1, kernel=kernel)
bluredImg = cv2.blur(img, ksize=(5, 5))
gaussianImg = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
medianImg = cv2.medianBlur(img, ksize=5)
bilateralImg = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# Display the results
titles = ['original', 'uniform kernel', 'uniform blur', 'gaussian blur', 'median blur', 'bilateral blur']
images = [img, uniformImg, bluredImg, gaussianImg, medianImg, bilateralImg]

for i in range(len(titles)):
    plt.subplot(3, 2, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
