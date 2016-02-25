'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
'''

import numpy as np
import cv2

# Load the image
img = cv2.imread('../data/opencv_logo.jpg')

# Convert it to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a median blur to remove noise
gray = cv2.medianBlur(gray, 3)

# Detect the circles
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

# Draw the circles on the original image
circles = np.uint16(np.around(circles))

for circle in circles[0, :]:
    # Draw the outer circle
    cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
    
    # Draw the circle center
    cv2.circle(img, (circle[0], circle[1]), 2, (0, 0, 255), 3)

# Display the results
cv2.imshow('Detected circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
