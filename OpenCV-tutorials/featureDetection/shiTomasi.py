'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
'''

import numpy as np
import cv2

# Load the image
img = cv2.imread('../images/chessboard.jpg')

# Convert it to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the Shi-Tomasi corners
gray = np.float32(gray)
corners = cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=0.01, minDistance=10)

# Paint the corners over the original image
for x, y in corners[:, 0]:
    cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
   
# Display the results
cv2.imshow('Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
