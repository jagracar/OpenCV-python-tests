'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
'''

import numpy as np
import cv2

# Start the webcam
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('../images/vtest.avi')

# Take the first frame and convert it to gray
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Create the HSV color image
hsvImg = np.zeros_like(frame)
hsvImg[..., 1] = 255

# Play until the user decides to stop
while True:
    # Save the previous frame data
    previousGray = gray
     
    # Get the next frame
    ret , frame = cap.read()
    
    if ret:
        # Convert the frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate the dense optical flow
        flow = cv2.calcOpticalFlowFarneback(previousGray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Obtain the flow magnitude and direction angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Update the color image
        hsvImg[..., 0] = 0.5 * ang * 180 / np.pi
        hsvImg[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgbImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
        
        # Display the resulting frame
        cv2.imshow('dense optical flow', np.hstack((frame, rgbImg)))
        k = cv2.waitKey(30) & 0xff
        
        # Exit if the user press ESC
        if k == 27:
            break
    else:
        break

# When everything is done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
