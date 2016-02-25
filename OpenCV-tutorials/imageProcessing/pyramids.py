'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html
'''

import cv2
import numpy as np, sys

# Load the two images that we want to blend
imgA = cv2.imread('../data/apple.jpg')
imgB = cv2.imread('../data/orange.jpg')

# Set the total number of pyramid levels
levels = 5

# Generate Gaussian pyramid for imgA
gaussianPyramidA = [imgA.copy()]
for i in range(1, levels):
    gaussianPyramidA.append(cv2.pyrDown(gaussianPyramidA[i - 1]))

# Generate Gaussian pyramid for imgB
gaussianPyramidB = [imgB.copy()]
for i in range(1, levels):
    gaussianPyramidB.append(cv2.pyrDown(gaussianPyramidB[i - 1]))

# Generate the inverse Laplacian Pyramid for imgA
laplacianPyramidA = [gaussianPyramidA[-1]]
for i in range(levels - 1, 0, -1):
    laplacian = cv2.subtract(gaussianPyramidA[i - 1], cv2.pyrUp(gaussianPyramidA[i]))
    laplacianPyramidA.append(laplacian)

# Generate the inverse Laplacian Pyramid for imgB
laplacianPyramidB = [gaussianPyramidB[-1]]
for i in range(levels - 1, 0, -1):
    laplacian = cv2.subtract(gaussianPyramidB[i - 1], cv2.pyrUp(gaussianPyramidB[i]))
    laplacianPyramidB.append(laplacian)

# Add the left and right halves of the Laplacian images in each level
laplacianPyramidComb = []
for laplacianA, laplacianB in zip(laplacianPyramidA, laplacianPyramidB):
    rows, cols, dpt = laplacianA.shape
    laplacianComb = np.hstack((laplacianA[:, 0:cols / 2], laplacianB[:, cols / 2:]))
    laplacianPyramidComb.append(laplacianComb)

# Reconstruct the image from the Laplacian pyramid
imgComb = laplacianPyramidComb[0]
for i in range(1, levels):
    imgComb = cv2.add(cv2.pyrUp(imgComb), laplacianPyramidComb[i])

# Display the result
cv2.imshow('image', imgComb)
cv2.waitKey(0)
cv2.destroyAllWindows()
