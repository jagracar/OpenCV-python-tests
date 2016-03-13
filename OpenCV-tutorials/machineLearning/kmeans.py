'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create two random 2D distributions with different mean values
x1 = np.random.randint(25, 50, (25, 2))
x2 = np.random.randint(60, 85, (25, 2))

# Combine the two distributions
x = np.vstack((x1, x2)).astype('float32')

# Apply KMeans
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(x, K=2, bestLabels=None, criteria=criteria, attempts=10, flags=flags)

# Separate the two groups
A = x[labels.ravel() == 0]
B = x[labels.ravel() == 1]

# Display the results
plt.hist(A[:, 0], bins=20, range=[20, 90], color='r')
plt.hist(B[:, 0], bins=20, range=[20, 90], color='b')
plt.scatter(A[:, 0], A[:, 1], color='r')
plt.scatter(B[:, 0], B[:, 1], color='b')
plt.scatter(centers[:, 0], centers[:, 1], s=80, color='y', marker='s')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()
