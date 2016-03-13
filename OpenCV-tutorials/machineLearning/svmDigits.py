'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html
'''

import cv2
import numpy as np

def deskew(img):
    '''
      Corrects the image skew
    '''
    # Calculate the image moments
    moments = cv2.moments(img)
    
    # Check if it's already fine
    if abs(moments['mu02']) < 1e-2:
        return img.copy()
    
    # Calculate the skew
    skew = moments['mu11'] / moments['mu02']
    
    # Correct the skew
    cellSize = int(np.sqrt(img.size))
    M = np.float32([[1, skew, -0.5 * cellSize * skew], [0, 1, 0]])
    flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (cellSize, cellSize), flags=flags)


def hog(img):
    '''
      Calculates the image Histograms of Oriented Gradients
    '''
    # Calculate the gradient images in polar coordinates
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(gx, gy)
    
    # Reduce the gradient angles to a fix number of values
    nBins = 16
    angle = np.int32(nBins * angle / (2 * np.pi))
    
    # Separate the gradient images in 4 cells
    angleCells = angle[:10, :10], angle[10:, :10], angle[:10, 10:], angle[10:, 10:]
    magnitudeCells = magnitude[:10, :10], magnitude[10:, :10], magnitude[:10, 10:], magnitude[10:, 10:]
    
    # Calculate the angle histograms for each cell, weighting the angle values with the magnitude values
    angleHists = [np.bincount(a.ravel(), m.ravel(), nBins) for a, m in zip(angleCells, magnitudeCells)]
    
    # Return the stack of the 4 cell histograms
    return np.hstack(angleHists)


# Load the file containing the digits data: 500 different images for each digit
data = cv2.imread('../data/digits.png', 0)

# Split the data in 5000 images, each of 20x20 pixels size
digitImages = [np.hsplit(row, 100) for row in np.vsplit(data, 50)]
print('digitImages dimensions:', str(len(digitImages)) + 'x' + str(len(digitImages[0])) + 'x' + str(digitImages[0][0].size))

# Deskew the digit images
digitImages = [list(map(deskew, row)) for row in digitImages]

# Calculate the images HOG histograms
hogHistograms = [list(map(hog, row)) for row in digitImages]

# Transform the Python array into a Numpy array
hogHistograms = np.float32(hogHistograms)
print('HOG histogram dimensions:', hogHistograms.shape[2])

# Use the hog histograms as descriptors for the train and test data sets
trainData = hogHistograms[:, :50].reshape(-1, hogHistograms.shape[2])
testData = hogHistograms[:, 50:].reshape(-1, hogHistograms.shape[2])

# Create the labels for the train and test data sets
digits = np.arange(10)
trainLabels = np.repeat(digits, trainData.shape[0] / digits.size)[:, np.newaxis]
testLabels = np.repeat(digits, testData.shape[0] / digits.size)[:, np.newaxis]

# Train the SVM
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)

# Test the result
ret, result = svm.predict(testData)

# Check the classification accuracy
correctMatches = np.count_nonzero(result == testLabels)
print('SVM classification accuracy:', 100.0 * correctMatches / result.size)
