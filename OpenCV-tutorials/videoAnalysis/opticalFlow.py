'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
'''

import numpy as np
import cv2

# Start the webcam
cap = cv2.VideoCapture(0)

# Take the first frame and convert it to gray
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Get the Shi Tomasi corners to use them as initial reference points
corners = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
cornerColors = np.random.randint(0, 255, (corners.shape[0], 3))

# Create a mask image for drawing purposes
mask = np.zeros_like(frame)

# Define the parameters for Lucas Kanade optical flow
lkParameters = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03))

# Play until the user decides to stop
while True:
    # Save the previous frame data
    previousGray = gray
    previousCorners = corners.reshape(-1, 1, 2)
    
    # Get the next frame
    ret , frame = cap.read()
    
    if ret:
        # Convert the frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        corners, st, err = cv2.calcOpticalFlowPyrLK(previousGray, gray, previousCorners, None, **lkParameters)
        
        # Select only the good corners
        corners = corners[st == 1]
        previousCorners = previousCorners[st == 1]
        cornerColors[st == 1]
        
        # Check that there are still some corners left
        if corners.shape[0] == 0:
            print('Stopping. There are no corners left to track')
            break
        
        # Draw the corner tracks
        for i in range(corners.shape[0]):
            x, y = corners[i]
            xPrev, yPrev = previousCorners[i]
            color = cornerColors[i].tolist()
            frame = cv2.circle(frame, (x, y), 5, color, -1)
            mask = cv2.line(mask, (x, y), (xPrev, yPrev), color, 2)
        frame = cv2.add(frame, mask)
        
        # Display the resulting frame
        cv2.imshow('optical flow', frame)
        k = cv2.waitKey(30) & 0xff
        
        # Exit if the user press ESC
        if k == 27:
            break
    else:
        break

# When everything is done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
