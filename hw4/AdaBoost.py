import csv
import numpy as np
import matplotlib.pyplot as plt
from DecisionStump import *


def read_data(filename):
    data_x = []
    data_y = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            data_x.append([float(line[0]), float(line[1])])
            data_y.append(int(line[-1]))
        f.close()
    return np.array(data_x), np.array(data_y)


class AdaBoost:
    def __init__(self):
        self.iteration = 0
        self.learners = []
        self.weighted_error = []
        self.votes = []
        self.weights = None
        self.y_pred = None
        self.accuracy = 0

    def fit(self, train_x, train_y, iteration=1):
        self.iteration = iteration
        self.weights = np.ones(train_y.shape[0])
        for k in range(iteration):
            model = DecisionStump()
            model.fit(train_x, train_y, self.weights)
            train_y_pred = model.predict(train_x)
            err = np.sum(self.weights * np.abs(train_y - train_y_pred) / 2) / np.sum(self.weights)
            vote = np.log((1 - err) / err) / 2
            self.weights = self.weights * np.exp(-train_y * vote * train_y_pred)  # update weights
            self.votes.append(vote)
            self.learners.append(model)

    def predict(self, test_x):
        s = np.zeros(test_x.shape[0])
        for k in range(self.iteration):
            s += self.votes[k] * self.learners[k].predict(test_x)
        self.y_pred = np.array(list(map(lambda x: -1 if x < 0 else 1, s / sum(self.votes))))
        return self.y_pred

    def eval_model(self, y, y_pred):
        self.accuracy = np.sum(np.abs(y + y_pred)) / (2 * y.shape[0])
        return self.accuracy

    def plot_result(self, train_x, train_y, test_x, test_y, iteration=1):
        plt.figure()
        accuracy = []
        self.iteration = 0
        self.weights = np.ones(train_y.shape[0])
        for k in range(iteration):
            self.iteration = k + 1
            model = DecisionStump()
            model.fit(train_x, train_y, self.weights)
            train_y_pred = model.predict(train_x)
            err = np.sum(self.weights * np.abs(train_y - train_y_pred) / 2) / np.sum(self.weights)
            vote = np.log((1 - err) / err) / 2
            self.weights = self.weights * np.exp(-train_y * vote * train_y_pred)  # update weights
            self.votes.append(vote)
            self.learners.append(model)
            test_y_pred = self.predict(test_x)
            self.eval_model(test_y, test_y_pred)
            accuracy.append(self.accuracy)
        plt.plot([i + 1 for i in range(iteration)], accuracy)
        plt.xlabel("Number of iterations")
        plt.ylabel("Accuracy")
        plt.title("Test accuracy verses number of iterations")
        plt.savefig("1.png")


if __name__ == "__main__":
    train_x, train_y = read_data("datasets/train_adaboost.csv")
    test_x, test_y = read_data("datasets/test_adaboost.csv")
    AB = AdaBoost()
    AB.plot_result(train_x, train_y, test_x, test_y, iteration=50)  # 5.1 plot
    print(AB.accuracy)  # 5.2 test accuracy after 50 iterations
    print(AB.learners[0].description)  # 5.3
    print(AB.learners[1].description)
    print(AB.learners[2].description)
