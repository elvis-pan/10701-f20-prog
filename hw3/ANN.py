"""
Artificial Neural Network
Author: Elvis Pan
Email: ypan2@andrew.cmu.edu
"""
from numba import jit, cuda
import csv
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

_EXP = 0  # large number to avoid exponentiation overflow


def read_data(filename):
    x = []
    y = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            x.append([float(d) for d in line[:-1]])
            vec = np.zeros(10)
            vec[int(float(line[-1]))] = 1
            y.append(vec)
        f.close()
    return np.array(x), np.array(y)


def read_params(filename):
    param = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data = [float(d) for d in line.strip('\n').split(',')]
            param.append(data)
        f.close()
    return np.array(param)


def save_model(Model, filename):
    with open(filename, "wb") as f:
        pickle.dump(Model, f, pickle.HIGHEST_PROTOCOL)


class ANN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        self.M = input_dim
        self.D = hidden_dim
        self.K = output_dim
        self.x = None
        self.a = None
        self.z = None
        self.b = None
        self.y = None
        self.y_pred = None
        self.loss = 0
        self.alpha = np.random.rand(self.D, self.M)
        self.alpha_bias = np.random.rand(self.D)
        self.beta = np.random.rand(self.K, self.D)
        self.beta_bias = np.random.rand(self.K)
        self.learning_rate = learning_rate

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        return np.exp(x - _EXP) / np.sum(np.exp(x - _EXP))

    @staticmethod
    def xeloss(pred, labels):
        return -np.sum(labels * np.log(pred))

    def forward(self, x, y):
        self.x = x.reshape((self.M, 1))
        self.y = y.reshape((self.K, 1))
        self.a = self.alpha_bias + np.matmul(self.alpha, self.x)
        self.z = self.sigmoid(self.a)
        self.b = self.beta_bias + np.matmul(self.beta, self.z)
        self.y_pred = self.softmax(self.b)
        self.loss = self.xeloss(self.y_pred, self.y)

    def backward(self):
        b_grad = self.y_pred - self.y
        beta_grad = np.matmul(b_grad, self.z.T)
        z_grad = np.matmul(self.beta.T, b_grad)
        a_grad = z_grad * self.z * (1 - self.z)
        alpha_grad = np.matmul(a_grad, self.x.T)
        self.beta -= self.learning_rate * beta_grad
        self.beta_bias -= self.learning_rate * b_grad
        self.alpha -= self.learning_rate * alpha_grad
        self.alpha_bias -= self.learning_rate * a_grad

    # @jit(target="cuda")
    def fit(self, train_x, train_y, epoch=1):
        for e in range(epoch):
            loss_total = 0
            for i in range(len(train_x)):
                x = train_x[i]
                y = train_y[i]
                self.forward(x, y)
                self.backward()
                loss_total += self.loss
            print(loss_total / len(train_x))

    def predict(self, test_x):
        res = []
        for x in test_x:
            x_mat = x.reshape((self.M, 1))
            a = self.alpha_bias + np.matmul(self.alpha, x_mat)
            z = self.sigmoid(a)
            b = self.beta_bias + np.matmul(self.beta, z)
            y_pred = self.softmax(b)
            res.append([x[0] for x in y_pred])
        return np.array(res)

    @staticmethod
    def compare(y, y_pred):
        cnt = 0
        y1 = np.argmax(y, axis=1)
        y2 = np.argmax(y_pred, axis=1)
        for i in range(len(y)):
            if y1[i] == y2[i]:
                cnt += 1
        return cnt / len(y)


if __name__ == "__main__":
    train_x, train_y = read_data("data/train.csv")
    test_x, test_y = read_data("data/test.csv")
    Model = ANN(784, 256, 10, learning_rate=0.01)
    Model.alpha = read_params("params/alpha1.txt")
    Model.alpha_bias = read_params("params/beta1.txt")
    Model.beta = read_params("params/alpha2.txt")
    Model.beta_bias = read_params("params/beta2.txt")
    start_time = time.time()
    Model.fit(train_x, train_y, epoch=100)
    train_y_pred = Model.predict(train_x)
    print("Training finished " + str(time.time() - start_time))
    save_model(Model, "model.pkl")
    print(Model.compare(train_y, train_y_pred))
    test_y_pred = Model.predict(test_x)
    print(Model.compare(test_y, test_y_pred))
    print("Finished" + str(time.time() - start_time))
