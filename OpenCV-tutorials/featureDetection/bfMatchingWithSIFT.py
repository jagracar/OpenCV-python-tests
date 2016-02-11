'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
'''

import numpy as np
import cv2

# Load the images in gray scale
img1 = cv2.imread('../images/box.png', 0)
img2 = cv2.imread('../images/box_in_scene.png', 0)

# Detect the SIFT key points and compute the descriptors for the two images
sift = cv2.xfeatures2d.SIFT_create()
keyPoints1, descriptors1 = sift.detectAndCompute(img1, None)
keyPoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Create brute-force matcher object
bf = cv2.BFMatcher()

# Match the descriptors
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
goodMatches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        goodMatches.append([m])

# Draw the first 10 matches
result = cv2.drawMatchesKnn(img1, keyPoints1, img2, keyPoints2, goodMatches, None, flags=2)

# Display the results
cv2.imshow('BF matches', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
