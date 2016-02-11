'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_watershed/py_watershed.html
'''

import numpy as np
import cv2

# Load the image
img = cv2.imread('../images/coins.jpg')

# Convert it to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image using Otsu’s binarization
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remove small regions
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Obtain the region that extends the foreground
extendedFg = cv2.dilate(thresh, kernel, iterations=2)

# Obtain the foreground centers
distance = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
ret, fgCenters = cv2.threshold(distance, 0.7 * distance.max(), 255, 0)

# Find the unknown regions subtrancting the foreground center to the extended foreground
fgCenters = np.uint8(fgCenters)
unknown = cv2.subtract(extendedFg, fgCenters)

# Define the markers
ret, markers = cv2.connectedComponents(fgCenters)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Mark the unknown region with zero
markers[unknown == 255] = 0

# Apply the watershed method
markers = cv2.watershed(img, markers)

# Draw the boundary regions, marked with -1, on the original image
img[markers == -1] = [255, 0, 0]

# Display the results
cv2.imshow('Separated coins', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
