'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the image in gray scale
img = cv2.imread('../data/wiki.jpg', 0)

# Calculate the histogram
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

# Calculate the cumulative distribution function of the histogram
cdf = hist.cumsum()

# Equalize the image
cdfMasked = np.ma.masked_equal(cdf, 0)
cdfMasked = (cdfMasked - cdfMasked.min()) * 255 / (cdfMasked.max() - cdfMasked.min())
equalizeFunction = np.ma.filled(cdfMasked, 0).astype('uint8')
equalizedImg = equalizeFunction[img]

# Calculate the histogram and the cdf of the equalized image
equalizedHist, bins = np.histogram(equalizedImg.flatten(), 256, [0, 256])
equalizedCdf = equalizedHist.cumsum()

# Display the results
plt.subplot(221)
plt.title('original')
plt.imshow(img, 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(222)
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.plot(cdf * hist.max() / cdf.max(), color='b')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')

plt.subplot(223)
plt.title('equalized')
plt.imshow(equalizedImg, 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(224)
plt.hist(equalizedImg.flatten(), 256, [0, 256], color='r')
plt.plot(equalizedCdf * equalizedHist.max() / equalizedCdf.max(), color='b')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')

plt.show()
