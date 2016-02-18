'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html
'''

import numpy as np
import cv2

# Initialize the video
cap = cv2.VideoCapture('../images/vtest.avi')

# Create the background subtraction object
method = 1

if method == 0:
    bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
elif method == 1:
    bgSubtractor = cv2.createBackgroundSubtractorMOG2()
else:
    bgSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG()

# Create the kernel that will be used to remove the noise in the foreground mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Play until the user decides to stop
while True:
    # Get the next frame
    ret , frame = cap.read()
    
    if ret:
        # Obtain the foreground mask
        foregroundMask = bgSubtractor.apply(frame)
        
        # Remove part of the noise
        foregroundMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_OPEN, kernel)
        
        # Display the mask
        cv2.imshow('background subtraction', foregroundMask)
        k = cv2.waitKey(30) & 0xff
        
        # Exit if the user press ESC
        if k == 27:
            break
    else:
        break

# When everything is done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
