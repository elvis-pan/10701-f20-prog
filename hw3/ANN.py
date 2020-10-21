"""
Artificial Neural Network
Author: Elvis Pan
Email: ypan2@andrew.cmu.edu
"""
import numpy as np
import matplotlib.pyplot as plt

class ANN:
    def __init__(self, M, D, K):
        self.M = M
        self.D = D
        self.K = K
        self.N = None
        self.x = None
        self.y = None
        self.y_pred = None
        self.alpha = np.zeros((D, M+1))
        self.beta = np.zeros((K, D+1))
        pass

    def linear_forward(self, x, weight, bias):
        pass

    def linear_backward(self):
        pass

    def sigmoid_forward(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_backward(self):
        pass

    def softmax_xeloss_forward(self, x, labels):
        return np.exp(x) / np.sum(np.exp(x))

    def softmax_xeloss_backward(self):
        pass

    def loss(self):
        return -1 / self.N * np.sum(self.y * np.log(self.y_pred))

    def train(self, x, y, epoch=1):
        self.N = len(x)
        self.x = x
        self.y = y
        for e in range(epoch):
            pass
        pass

    def predict(self, x):
        pass
