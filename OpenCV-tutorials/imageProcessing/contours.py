'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html
'''

import numpy as np
import cv2

# Create a black image
imgWidth = 500
imgHeight = 300
img = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)

# Draw a blue rectangle
startPoint = (int(imgWidth / 4), int(imgHeight / 4))
endPoint = (int(3 * imgWidth / 4), int(3 * imgHeight / 4))
cv2.rectangle(img, startPoint, endPoint, color=(255, 0, 0), thickness=-1)

# Draw a black rectangle inside
startPoint = (int(imgWidth / 3), int(imgHeight / 3))
endPoint = (int(2 * imgWidth / 3), int(2 * imgHeight / 3))
cv2.rectangle(img, startPoint, endPoint, color=(0, 0, 0), thickness=-1)

# Rotate the image
M = cv2.getRotationMatrix2D((imgWidth / 2, imgHeight / 2), 40, 1)
img = cv2.warpAffine(img, M, (imgWidth, imgHeight))

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a binary image
ret, thresh = cv2.threshold(gray, 20, 255, 0)

# Obtain the image contours
thresh, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# Draw the points from the outer contour
cnt = contours[0]
for point in cnt.reshape(-1, 2):
    cv2.circle(img, tuple(point), 7, color=(0, 0, 255), thickness=2)

# Calculate the centroid
M = cv2.moments(cnt)
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
cv2.circle(img, (cx, cy), 7, color=(255, 255, 255), thickness=-1)

# Print some other information
print('Contour area:', cv2.contourArea(cnt, True))
print('Contour perimeter:', cv2.arcLength(cnt, True))

# Simplify the contour
epsilon = 0.1 * cv2.arcLength(cnt, True)
cntApprox = cv2.approxPolyDP(cnt, epsilon, True)
for point in cntApprox.reshape(-1, 2):
    cv2.circle(img, tuple(point), 7, color=(255, 0, 0), thickness=2)

# Draw the bounding rectangle
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

# Draw the minimum area rectangle
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img, [box], 0, (0, 255, 255), 2)

# Draw the minimum enclosing circle
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
cv2.circle(img, center, radius, (255, 0, 255), 2)

# Draw an ellipse around it
ellipse = cv2.fitEllipse(cnt)
cv2.ellipse(img, ellipse, (255, 255, 255), 2)
rows, cols = img.shape[:2]

# Fit a line to the contour and draw it
[vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)
cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

# Get the extreme points
leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
cv2.circle(img, leftmost, 4, color=(255, 0, 0), thickness=-1)
cv2.circle(img, rightmost, 4, color=(255, 0, 0), thickness=-1)
cv2.circle(img, topmost, 4, color=(255, 0, 0), thickness=-1)
cv2.circle(img, bottommost, 4, color=(255, 0, 0), thickness=-1)

# Display the result
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
