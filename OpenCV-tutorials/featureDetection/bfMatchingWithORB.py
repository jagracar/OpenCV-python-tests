'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
'''

import numpy as np
import cv2

# Load the images in gray scale
img1 = cv2.imread('../data/box.png', 0)
img2 = cv2.imread('../data/box_in_scene.png', 0)

# Detect the ORB key points and compute the descriptors for the two images
orb = cv2.ORB_create()
keyPoints1, descriptors1 = orb.detectAndCompute(img1, None)
keyPoints2, descriptors2 = orb.detectAndCompute(img2, None)

# Create brute-force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match the descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort them in by distance
matches = sorted(matches, key=lambda x:x.distance)

# Draw the first 10 matches
result = cv2.drawMatches(img1, keyPoints1, img2, keyPoints2, matches[:10], None, flags=2)

# Display the results
cv2.imshow('BF matches', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
