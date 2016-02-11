'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html
'''

import numpy as np
import cv2

# Load the image
img = cv2.imread('../images/home.jpg')

# Convert it to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the SHIFT key points
sift = cv2.xfeatures2d.SIFT_create()
keyPoints, descriptors = sift.detectAndCompute(gray, None)

# Paint the key points over the original image
result = cv2.drawKeypoints(img, keyPoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the results
cv2.imshow('Key points', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
