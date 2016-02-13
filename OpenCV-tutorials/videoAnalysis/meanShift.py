'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_meanshift/py_meanshift.html
'''

import numpy as np
import cv2

# Start the webcam
cap = cv2.VideoCapture(0)

# Take the first frame
ret, frame = cap.read()
rows, cols = frame.shape[:2]

# Define the initial window location at the frame center
windowWidth = 150
windowHeight = 200
windowCol = int((cols - windowWidth) / 2)
windowRow = int((rows - windowHeight) / 2)
window = (windowCol, windowRow, windowWidth, windowHeight)

# Get the ROI and convert it to HSV scale
roi = frame[windowRow:windowRow + windowHeight, windowCol:windowCol + windowWidth]
roiHsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Mask the dark areas
lowLimit = np.array((0., 60., 32.))
highLimit = np.array((180., 255., 255.))
mask = cv2.inRange(roiHsv, lowLimit, highLimit)

# Calculate the hue histogram of the unmasked region
roiHist = cv2.calcHist([roiHsv], [0], mask, [180], [0, 180])
cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)

# Set the termination criteria: either finished 10 iteration or moved less than one pixel
terminationCriteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS , 10, 1)

# Play until the user decides to stop
while True:
    # Get the next frame
    ret , frame = cap.read()

    if ret:
        # Calculate the histogram back projection
        frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backprojectedFrame = cv2.calcBackProject([frameHsv], [0], roiHist, [0, 180], 1)

        # Mask the dark areas to improve the results
        mask = cv2.inRange(frameHsv, lowLimit, highLimit)
        backprojectedFrame &= mask

        # Apply meanshift method to get the new window location
        ret, window = cv2.meanShift(backprojectedFrame, window, terminationCriteria)

        # Draw the window on the frame
        windowCol, windowRow = window[:2]
        frame = cv2.rectangle(frame, (windowCol, windowRow), (windowCol + windowWidth, windowRow + windowHeight), 255, 2)

        # Display the resulting frame
        cv2.imshow('meanshift', frame)
        k = cv2.waitKey(60) & 0xff
        
        # Exit if the user press ESC
        if k == 27:
            break
    else:
        break

# When everything is done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
