'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
'''

import numpy as np
import cv2

# Load the image
img = cv2.imread('../images/sudoku-original.jpg')
rows, cols, channels = img.shape

# Convert it to gray scale and detect the edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

# Obtain the Hough transform
deltaRho = 1
deltaTheta = np.pi / 180
threshold = 200
lines = cv2.HoughLines(edges, deltaRho, deltaTheta, threshold)

# Paint the lines on the image
maxLength = np.sqrt(rows ** 2 + cols ** 2)

for rho, theta in lines[:, 0]:
    cos = np.cos(theta)
    sin = np.sin(theta)
    x0 = rho * cos
    y0 = rho * sin
    x1 = int(x0 + maxLength * sin)
    y1 = int(y0 - maxLength * cos)
    x2 = int(x0 - maxLength * sin)
    y2 = int(y0 + maxLength * cos)
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

# Obtain the probabilistic Hough transform
deltaRho = 1
deltaTheta = np.pi / 180
threshold = 100
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges, deltaRho, deltaTheta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

# Paint the lines on the image
for x1, y1, x2, y2 in lines[:, 0]:
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

# Display the results
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
