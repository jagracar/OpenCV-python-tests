'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html
'''

import numpy as np
import cv2

# Create a black image
imgWidth = 500
imgHeight = 300
img = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)

# Draw a diagonal blue line
cv2.line(img, (0, 0), (imgWidth - 1, imgHeight - 1), color=(255, 0, 0), thickness=5)

# Draw a green rectangle
cv2.rectangle(img, (int(imgWidth / 2), 10), (imgWidth, int(imgHeight / 2)), color=(0, 255, 0), thickness=3)

# Draw a red circle
cv2.circle(img, (int(imgWidth / 2), int(imgHeight / 2)), 50, color=(0, 0, 255), thickness=-1)

# Draw an inclined ellipse
cv2.ellipse(img, (100, imgHeight - 100), axes=(100, 50), angle=20, startAngle=0, endAngle=180, color=255, thickness=-1)

# Draw a closed polygon defined by a set of points
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], dtype=np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255))

# Finally draw some text with anti-aliased font
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV', (10, 250), fontFace=font, fontScale=2, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

# Display the final image
cv2.imshow('image', img)
     
# Exit when a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
