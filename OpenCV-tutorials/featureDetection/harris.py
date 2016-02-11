'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
'''

import numpy as np
import cv2

# Load the image
img = cv2.imread('../images/chessboard.jpg')

# Convert it to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the Harris corners
gray = np.float32(gray)
cornersImg = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Dilate the image to increase the corners size
cornersImg = cv2.dilate(cornersImg, np.ones((3, 3)))

# Paint the corners over the original image
img[cornersImg > 0.2 * cornersImg.max()] = [0, 0, 255]

# Display the results
cv2.imshow('Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
