'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_knn/py_knn_opencv/py_knn_opencv.html
'''

import cv2
import numpy as np

# Load the file containing the digits data: 500 different images for each digit
data = cv2.imread('../data/digits.png', 0)

# Split the data in 5000 images, each of 20x20 pixels size
digitImages = [np.hsplit(row, 100) for row in np.vsplit(data, 50)]
print('digitImages dimensions:', str(len(digitImages)) + 'x' + str(len(digitImages[0])) + 'x' + str(digitImages[0][0].size))

# Transform the Python array into a Numpy array
digitImages = np.float32(digitImages)

# Use the image pixel values as descriptors for the train and test data sets
trainData = digitImages[:, :50].reshape(-1, digitImages[0, 0].size)
testData = digitImages[:, 50:].reshape(-1, digitImages[0, 0].size)

# Create the labels for the train and test data sets
digits = np.arange(10)
trainLabels = np.repeat(digits, trainData.shape[0] / digits.size)[:, np.newaxis]
testLabels = np.repeat(digits, testData.shape[0] / digits.size)[:, np.newaxis]

# Save the train data
np.savez('../data/knn_data.npz', trainData=trainData, trainLabels=trainLabels)

# Load the train data
with np.load('../data/knn_data.npz') as data:
    trainData = data['trainData']
    trainLabels = data['trainLabels']

# Train the kNN
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)

# Test the kNN
ret, result, neighbours, dist = knn.findNearest(testData, k=5)

# Check the classification accuracy
correctMatches = np.count_nonzero(result == testLabels)
print('kNN classification accuracy:', 100.0 * correctMatches / result.size)
