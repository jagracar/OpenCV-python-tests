'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_optimization/py_optimization.html
'''

import numpy as np
import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Get the next frame
    ret, frame = cap.read()

    # Convert from BGR scale to HSV scale
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of blue color in HSV
    lowerBlue = np.array([110, 50, 50])
    upperBlue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lowerBlue, upperBlue)

    # Remove everything that is not in the mask
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Display all the difference images
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    
    # User interaction
    if k == 27:
        break

# When everything is done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
