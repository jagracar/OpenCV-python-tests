'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
'''

import numpy as np
import cv2

# Load the face and eye cascade classifiers
faceCascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Display the capture frames
while True:
    # Get the next frame
    ret, frame = cap.read()

    if ret:
        # Convert the frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for x, y, w, h in faces:
            # Draw the face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Obtain the region of interest and detect the eyes
            roi = gray[y:y + h, x:x + w]
            eyes = eyeCascade.detectMultiScale(roi)
            eyeCounter = 0
            
            for ex, ey, ew, eh in eyes:
                # Draw only the first two eyes
                if eyeCounter > 1:
                    break
                
                # Draw the eye rectangle
                cv2.rectangle(frame, (ex + x, ey + y), (ex + x + ew, ey + y + eh), (0, 255, 0), 2)
                eyeCounter += 1
        
        # Display the resulting frame
        cv2.imshow('frame', frame)
        k = cv2.waitKey(20) & 0xFF
        
        # User interaction
        if k == ord('q'):
            break
    else:
        break

# When everything is done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
