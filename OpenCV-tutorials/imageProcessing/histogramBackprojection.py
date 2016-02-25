'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_histograms/py_histogram_backprojection/py_histogram_backprojection.html
'''

import numpy as np
import cv2

# Load the target image
target = cv2.imread('../data/rose.png')
targetHsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

# Define the model image
model = target[100:170, 300:400]
modelHsv = cv2.cvtColor(model, cv2.COLOR_BGR2HSV)

# Calculate the model 2D histogram
modelHist = cv2.calcHist([modelHsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
maxHistValue = np.max(modelHist)

# Create a new backprojected image using the model histogram 
h, s, v = cv2.split(targetHsv)
probability = modelHist[h.ravel(), s.ravel()] / maxHistValue
backprojectedImg = np.uint8(255 * probability).reshape(target.shape[:2])

# Close small holes
disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
backprojectedImg = cv2.morphologyEx(backprojectedImg, op=cv2.MORPH_CLOSE, kernel=disk)

# Threshold the backprojected image
ret, thresh = cv2.threshold(backprojectedImg, 1, 255, 0)
thresh = cv2.merge((thresh, thresh, thresh))

# Obtain the final filtered image
result = cv2.bitwise_and(target, thresh)

# Display the results
combined1 = np.hstack((target, result))
combined2 = np.hstack((cv2.merge((backprojectedImg, backprojectedImg, backprojectedImg)), thresh))
combined = np.vstack((combined1, combined2))
cv2.imshow('result', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
