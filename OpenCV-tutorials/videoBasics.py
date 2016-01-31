'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
'''

import numpy as np
import cv2

# Initialize the video capture
useWebcam = True

if useWebcam:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture('images/drop.avi')

# Check that the capture is open
if not cap.isOpened():
    cap.open()

# Get some useful information
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(5))

if fps == -1:
    fps = 30

# Define the video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoOutput = cv2.VideoWriter('out/output.avi', fourcc, fps, (width, height))

#
# Display the capture frames
while True:
    # Get the next frame
    ret, frame = cap.read()

    if ret:
        # Convert the frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Flip the frame
        gray = cv2.flip(gray, 1)
        
        # Convert the frame again to BGR
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Write the flipped gray frame
        videoOutput.write(gray)
        
        # Display the resulting frame
        cv2.imshow('frame', gray)
        k = cv2.waitKey(round(1000 / fps)) & 0xFF
        
        # User interaction
        if k == ord('q'):
            break
    else:
        break

# When everything is done, release the capture and close all windows
cap.release()
videoOutput.release()
cv2.destroyAllWindows()
