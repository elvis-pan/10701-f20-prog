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
    return standardize(data)


def standardize(data):
    data_arr = np.array(data)
    data_res = []
    for line in data_arr.T:
        data_res.append((line - np.mean(line)) / np.std(line))
    return np.array(data_res).T


def split_data(data):
    X = [line[1:] for line in data]
    y = [line[0] for line in data]
    return np.array(X), np.array(y)


class LinearRegression:
    def __init__(self, num_feature, learn_rate, reg_type=None, reg=0):
        self.m = num_feature
        self.eta = learn_rate
        self.w = np.zeros(num_feature)
        self.b = 0.0
        self.reg_type = reg_type
        self.lambda_ = reg

    def reg(self):
        if self.reg_type is None:
            return 0
        if self.reg_type == "Ridge":
            return self.lambda_ * np.pow(self.w, 2)
        if self.reg_type == "Lasso":
            return self.lambda_ * np.abs(self.w)

    def fit(self, X, y, epoch=1):
        for e in range(epoch):
            for i in range(X.shape[0]):
                y_pred = np.dot(self.w, X[i]) + self.b
                self.w = self.w - 2 * self.eta * (y_pred - y[i]) * X[i] + self.reg()
                self.b = self.b - 2 * self.eta * (y_pred - y[i])
                print(self.w)
                print(self.b)
        return self.w, self.b

    def predict(self, X):
        return np.dot(X, self.w.T) + self.b


if __name__ == "__main__":
    train_data = read_data("carseats_train.csv")
    test_data = read_data("carseats_test.csv")
    X_train, y_train = split_data(train_data)

    LR = LinearRegression(12, 0.01)
    LR.fit(X_train, y_train)
    print(LR.w)
    LR.predict(X_train)
