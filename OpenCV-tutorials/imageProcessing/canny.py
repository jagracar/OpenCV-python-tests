'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_canny/py_canny.html
'''

import numpy as np
import cv2

# Define the trackbars call back functions
def threshold1Callback(x):
    global threshold1, edges
    threshold1 = x
    edges = cv2.Canny(img, threshold1, threshold2)
    return

def threshold2Callback(x):
    global threshold2, edges
    threshold2 = x
    edges = cv2.Canny(img, threshold1, threshold2)
    return

# Load the image in gray scale
img = cv2.imread('../images/messi5.jpg', 0)

# Apply the Canny edge detection algorithm with the initial threshold values
threshold1 = 100
threshold2 = 200
edges = cv2.Canny(img, threshold1, threshold2)

# Create the display window and add the two trackbars
cv2.namedWindow('canny')
cv2.createTrackbar('threshold1', 'canny', threshold1, 255, threshold1Callback)
cv2.createTrackbar('threshold2', 'canny', threshold2, 255, threshold2Callback)

# Display the results
while True:
    cv2.imshow('canny', edges)
    k = cv2.waitKey(1) & 0xFF
    
    # Exit if the user presses the ESC key
    if k == 27:
        break

# Destroy all windows
cv2.destroyAllWindows()
