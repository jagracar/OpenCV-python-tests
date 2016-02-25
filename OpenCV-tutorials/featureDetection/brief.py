'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_brief/py_brief.html
'''

import numpy as np
import cv2

# Load the image
img = cv2.imread('../data/simple.jpg')

# Convert it to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the CenSurE key points
star = cv2.xfeatures2d.StarDetector_create()
keyPoints = star.detect(gray, None)

# Create the BRIEF extractor and compute the descriptors
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
keyPoints, descriptors = brief.compute(img, keyPoints)

# Paint the key points over the original image
result = cv2.drawKeypoints(img, keyPoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the results
cv2.imshow('Key points', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
