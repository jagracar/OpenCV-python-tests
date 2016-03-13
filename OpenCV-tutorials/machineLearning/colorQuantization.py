'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('../data/home.jpg')

# Use the colors as the descriptors
x = img.reshape(-1, 3).astype('float32')

# Apply KMeans
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(x, K=6, bestLabels=None, criteria=criteria, attempts=10, flags=flags)

# Create the new image
centers = np.uint8(centers)
newImage = centers[labels.ravel()]
newImage = newImage.reshape(img.shape)

# Display the results
cv2.imshow('New image', newImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
