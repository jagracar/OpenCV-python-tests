'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_knn/py_knn_opencv/py_knn_opencv.html
'''

import cv2
import numpy as np

# Load the alphabet data
data = np.loadtxt('../data/letter-recognition.data', dtype='float32', delimiter=',', converters={0: lambda ch: ord(ch) - ord('A')})

# Split the data in train data and test data
trainData, testData = np.vsplit(data, 2)

# split trainData and testData to features and responses
trainLabels, trainData = np.hsplit(trainData, [1])
testLabels, testData = np.hsplit(testData, [1])

# Train the kNN
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)

# Test the kNN
ret, result, neighbours, dist = knn.findNearest(testData, k=5)

# Check the classification accuracy
correctMatches = np.count_nonzero(result == testLabels)
print('kNN classification accuracy:', 100.0 * correctMatches / result.size)
