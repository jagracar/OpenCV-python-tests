'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_knn/py_knn_understanding/py_knn_understanding.html
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set the training data (x, y) coordinates
trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)

# Split the training data in two groups
responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)
redGroup = trainData[responses.ravel() == 0]
blueGroup = trainData[responses.ravel() == 1]

# Train the kNN
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)

# Create some new data and classify it
newData = np.random.randint(0, 100, (1, 2)).astype(np.float32)
ret, results, neighbours , dist = knn.findNearest(newData, 3)

print("result: ", results)
print("neighbours: ", neighbours)
print("distance: ", dist)

# Display the results
plt.scatter(redGroup[:, 0], redGroup[:, 1], 80, 'r', '^')
plt.scatter(blueGroup[:, 0], blueGroup[:, 1], 80, 'b', 's')
plt.scatter(newData[:, 0], newData[:, 1], 200, 'r' if results[0] == 0 else 'b', 'o')
plt.scatter(newData[:, 0], newData[:, 1], 80, 'g', 'o')
plt.show()
