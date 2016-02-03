'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_trackbar/py_trackbar.html
'''

import cv2
import numpy as np

# Create a black image
img = np.zeros((300, 512, 3), np.uint8)

# Define the trackbars call back function
def nothing(x):
    pass

# Create a window and add the trackbars to it
cv2.namedWindow('image')
cv2.createTrackbar('R', 'image', 100, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)

# Add also a switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image', 1, 1, nothing)

while True:
    # Display the image
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    
    # Exit if the user presses a key
    if k == 27:
        break

    # Update the image depending on the trackbar values
    if cv2.getTrackbarPos(switch, 'image'):
        r = cv2.getTrackbarPos('R', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        img[:] = [b, g, r]
    else:
        img[:] = 0

# Destroy all windows
cv2.destroyAllWindows()
