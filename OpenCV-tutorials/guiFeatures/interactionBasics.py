'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html
'''

import numpy as np
import cv2

# Print all the available events
for i in dir(cv2):
    if 'EVENT' in i:
        print(i)

# Define the mouse event callback function
xInit, yInit = -1, -1
drawing = False
drawRectangle = True

def drawCircle(event, x, y, flags, param):
    # Global variables that we want to update
    global xInit, yInit, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        xInit, yInit = x, y
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:  #
            drawFigure(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawFigure(x, y)
        drawing = False

def drawFigure(x, y):
    if drawRectangle == True:
        cv2.rectangle(img, (xInit, yInit), (x, y), (0, 255, 0), -1)
    else:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)


# Create a black image
img = np.zeros((512, 512, 3), np.uint8)

# Create the display window and bind the mouse event function to it
cv2.namedWindow('image')
cv2.setMouseCallback('image', drawCircle)

# Update the window until the user decides to exit
while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(20) & 0xFF
    
    if k == ord('m'):
        drawRectangle = not drawRectangle
    elif k == 27 or k == ord('q'):
        break

# Destroy all windows before exit
cv2.destroyAllWindows()
