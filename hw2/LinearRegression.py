import csv
import numpy as np


def read_data(filename):
    data = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for line in reader:
            data_line = [float(k) for k in line[:6]]
            if line[6] == 'Good':
                data_line.extend((1.0, 0.0, 0.0))
            elif line[6] == 'Medium':
                data_line.extend((0.0, 1.0, 0.0))
            elif line[6] == 'Bad':
                data_line.extend((0.0, 0.0, 1.0))
            data_line.extend([float(k) for k in line[7:9]])
            if line[9] == "Yes":
                data_line.append(1.0)
            else:
                data_line.append(0.0)
            if line[10] == "Yes":
                data_line.append(1.0)
            else:
                data_line.append(0.0)
            data.append(data_line)
        f.close()
    return data


def split_data(data):
    X = [line[1:] for line in data]
    y = [line[0] for line in data]
    return np.array(X), np.array(y)


def compute_error(y_pred, y):
    return np.mean(np.power(y_pred - y, 2))


class LinearRegression:
    def __init__(self, num_feature, learn_rate, reg_type=None, reg=0):
        self.m = num_feature
        self.eta = learn_rate
        self.w = np.zeros(num_feature)
        self.b = 0.0
        self.reg_type = reg_type
        self.lambda_ = reg
        self.X = None
        self.y = None
        self.mean = None
        self.std = None
        self.error = []

    def find_params(self, X):
        self.mean = np.array([np.mean(line) for line in X.T])
        self.std = np.array([np.std(line) for line in X.T])

    def standardize(self, X):
        res = []
        for i in range(X.T.shape[0]):
            res.append((X.T[i] - self.mean[i]) / self.std[i])
        return np.array(res).T

    def reg(self):
        if self.reg_type is None:
            return 0
        if self.reg_type == "Ridge":
            return self.lambda_ * np.power(self.w, 2)
        if self.reg_type == "Lasso":
            return self.lambda_ * np.abs(self.w)

    def fit(self, X, y, epoch=1):
        self.find_params(X)
        self.X = self.standardize(X)
        self.y = y
        for e in range(epoch):
            for i in range(self.X.shape[0]):
                y_pred = np.dot(self.w, self.X[i]) + self.b
                self.w = self.w - 2 * self.eta * (y_pred - self.y[i]) * self.X[i] + self.reg()
                self.b = self.b - 2 * self.eta * (y_pred - self.y[i])
            self.error.append(compute_error(self.predict(X), y))
        return self.w, self.b

    def predict(self, X):
        X_standardized = self.standardize(X)
        return np.dot(X_standardized, self.w.T) + self.b


if __name__ == "__main__":
    train_data = read_data("carseats_train.csv")
    test_data = read_data("carseats_test.csv")
    X_train, y_train = split_data(train_data)
    X_test, y_test = split_data(test_data)

    LR = LinearRegression(12, 0.01)
    LR.fit(X_train, y_train, epoch=50)
    y_pred_train = LR.predict(X_train)

    y_pred_test = LR.predict(X_test)
    print(compute_error(y_pred_test, y_test))
