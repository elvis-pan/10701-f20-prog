from ANN import *
import numpy as np

Model = ANN(6, 4, 3, learning_rate=1, regularization=0.01)
x = np.array([[1, 1, 0, 0, 1, 1]])
y = np.array([[0, 1, 0]])
Model.alpha = np.array([[1, 1, -1, -1, 0, -1], [3, 1, 0, 1, 0, 2], [1, 2, -1, 0, 2, -1], [2, 0, 2, 1, -2, 1]],
                       dtype=float)
Model.beta = np.array([[3, -1, 2, 1], [1, -1, 2, 2], [1, -1, 1, 1]], dtype=float)
Model.fit(x, y, epoch=1)
print("Finished")
