"""
Artificial Neural Network
Author: Elvis Pan
Email: ypan2@andrew.cmu.edu
"""
import numpy as np
import matplotlib.pyplot as plt


class ANN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.M = input_dim
        self.D = hidden_dim
        self.K = output_dim
        self.alpha = np.zeros((self.D, self.M))
        self.alpha_bias = np.zeros(self.D)
        self.beta = np.zeros((self.K, self.D))
        self.beta_bias = np.zeros(self.K)
        pass

    @staticmethod
    def linear_forward(x, weight, bias):
        return bias + np.dot(weight, x)

    def linear_backward(self):
        pass

    @staticmethod
    def sigmoid_forward(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_backward(self):
        pass

    @staticmethod
    def softmax_xeloss_forward(x, labels):
        pred = np.exp(x) / np.sum(np.exp(x))
        return -np.sum(labels * np.log(pred))

    def softmax_xeloss_backward(self):
        pass

    def forward(self, x):
        out = self.linear_forward(x, self.alpha, self.alpha_bias)
        out = self.sigmoid_forward(out)
        out = self.linear_forward(out, self.beta, self.beta_bias)
        out = self.softmax_xeloss_forward(out)
        return out

    def backward(self):
        pass

    def train(self, x, y, epoch=1):
        for e in range(epoch):
            pass
        pass

    def predict(self, x):
        pass
