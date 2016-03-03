'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html
'''

import cv2
import numpy as np


cellSize = 20
nBins = 16

# Corrects the image skew
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * cellSize * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (cellSize, cellSize), flags=affine_flags)
    return img

# Calculates the HOG histograms
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(nBins * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), nBins) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  # hist is a 64 bit vector
    return hist

# Load the image containing the training data
img = cv2.imread('../data/digits.png', 0)

# Split the image in 5000 cells, each of 20x20 pixels size
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

# Split the cells into train cells and test cells
trainCells = [ i[:50] for i in cells ]
testCells = [ i[50:] for i in cells]

######     Now training      ########################

svm_params = dict(kernel_type=cv2.ml.SVM_LINEAR, svm_type=cv2.ml.SVM_C_SVC, C=2.67, gamma=5.383)

affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR

trainCellsDeskewed = [list(map(deskew, row)) for row in trainCells]
trainHogData = [list(map(hog, row)) for row in trainCellsDeskewed]
trainData = np.float32(trainHogData).reshape(-1, 64)
trainLabels = np.float32(np.repeat(np.arange(10), 250)[:, np.newaxis])

svm = cv2.ml.SVM_create()
svm.train(trainData, trainLabels, params=svm_params)
svm.save('svm_data.dat')

######     Now testing      ########################

testCellsDeskewed = [list(map(deskew, row)) for row in testCells]
testHogData = [list(map(hog, row)) for row in testCellsDeskewed]
testData = np.float32(testHogData).reshape(-1, nBins * 4)
result = svm.predict_all(testData)

# Check the classification accuracy
correctMatches = np.count_nonzero(result == testLabels)
print('SVM classification accuracy:', 100.0 * correctMatches / result.size)
