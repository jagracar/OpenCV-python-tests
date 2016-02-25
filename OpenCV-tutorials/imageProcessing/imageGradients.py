'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the image in gray scale
img = cv2.imread('../data/sudoku-original.jpg', 0)

# Calculate the different filters
laplacian = cv2.Laplacian(img, ddepth=cv2.CV_64F)
sobelx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)

# Remove the negative values taking the absolute
laplacian = np.absolute(laplacian)
sobelx = np.absolute(sobelx)
sobely = np.absolute(sobely)

# Display the results
titles = ['original', 'Laplacian', 'Sobel x', 'Sobel y']
images = [img, laplacian, sobelx, sobely]

for i in range(len(titles)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
