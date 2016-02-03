'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load an image
img = cv2.imread('../images/messi5.jpg', 0)  # in greyscale format
# img = cv2.imread('../images/messi5.jpg', -1)  # in bgr format

# Display the image using matplotlib or the opencv methods
useMatplotlib = True

if(useMatplotlib):
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  
    plt.show()
else:
    # Create an empty window
    windowName = 'image' 
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    
    # Display the image on the window
    cv2.imshow(windowName, img)
    
    # Wait until a key is pressed
    k = cv2.waitKey(0) & 0xFF
    
    if k == 27:
        # Close all windows
        cv2.destroyAllWindows()
        # cv2.destroyWindow(windowName)
    elif k == ord('s'):
        # Save the modified image
        cv2.imwrite('../out/messigray.png', img)
        cv2.destroyAllWindows()

