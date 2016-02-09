'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the images in grey scale
originalImg = cv2.imread('../images/messi5.jpg', 0)
template = cv2.imread('../images/messi_face.jpg', 0)
w, h = template.shape[::-1]

# Compare all the methods
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for method in methods:
    # Math the template
    methodCode = eval(method)
    result = cv2.matchTemplate(originalImg, template, methodCode)
    minValue, maxValue, minLoc, maxLoc = cv2.minMaxLoc(result)

    # If the methodCode is TM_SQDIFF or TM_SQDIFF_NORMED, take the minimum position
    if methodCode in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        topLeftCorner = minLoc
    else:
        topLeftCorner = maxLoc
    bottomRightCorner = (topLeftCorner[0] + w, topLeftCorner[1] + h)

    # Draw the square in a copy of the original image
    img = originalImg.copy()
    cv2.rectangle(img, topLeftCorner, bottomRightCorner, 255, 2)

    # Display the results
    plt.subplot(121)
    plt.imshow(result, cmap='gray')
    plt.title('Matching Result')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(122)
    plt.imshow(img, cmap='gray')
    plt.title('Detected Point')
    plt.xticks([])
    plt.yticks([])
    
    plt.suptitle(method)
    plt.show()
