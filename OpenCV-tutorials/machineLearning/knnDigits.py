'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_knn/py_knn_opencv/py_knn_opencv.html
'''

import cv2
import numpy as np

# Load the image containing the training data
img = cv2.imread('../data/digits.png', 0)

# Split the image in 5000 cells, each of 20x20 pixels size
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

# Transform the python array into a Numpy array
cells = np.array(cells)

# Split the data into train data and test data
trainData = cells[:, :50].reshape(-1, 400).astype(np.float32)
testData = cells[:, 50:100].reshape(-1, 400).astype(np.float32)

# Create the labels for the train and test data
digits = np.arange(10)
trainLabels = np.repeat(digits, 250)[:, np.newaxis]
testLabels = trainLabels.copy()

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
