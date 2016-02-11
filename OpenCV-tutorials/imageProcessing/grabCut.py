'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html
'''

import numpy as np
import cv2

# Load the image
img = cv2.imread('../images/messi5.jpg')

# Run the grab cut algorithm
mask = np.zeros(img.shape[:2], np.uint8)
rect = (50, 50, 450, 290)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

# Combine the secure regions with the probable ones
resultMask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * resultMask[:, :, np.newaxis]

# Display the results
cv2.imshow('Separated coins', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
