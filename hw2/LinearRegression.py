import csv
import numpy as np
import matplotlib.pyplot as plt


def split_data(data):
    X = np.array([line[1:] for line in data])
    y = np.array([line[0] for line in data])
    return X, y


def read_data(filename):
    data = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # do not read header
        for line in reader:
            data_line = [float(k) for k in line[:6]]
            data_line.extend([float(k) for k in line[7:9]])
            if line[9] == "Yes":
                data_line.append(1.0)
            else:
                data_line.append(0.0)
            if line[10] == "Yes":
                data_line.append(1.0)
            else:
                data_line.append(0.0)
            if line[6] == 'Good':
                data_line.extend((0.0, 1.0, 0.0))
            elif line[6] == 'Medium':
                data_line.extend((0.0, 0.0, 1.0))
            elif line[6] == 'Bad':
                data_line.extend((1.0, 0.0, 0.0))
            data.append(data_line)
        f.close()
    return split_data(data)


def compute_error(y_pred, y):
    return np.mean(np.power(y_pred - y, 2))


class LinearRegression:
    def __init__(self, num_feature, learn_rate, regularization=None, penalty=0):
        self.m = num_feature
        self.eta = learn_rate
        self.w = np.zeros(num_feature, dtype=np.float64)
        self.b = 0.0
        self.reg_type = regularization
        self.lambda_ = penalty
        self.epoch = 1
        self.X = None
        self.y = None
        self.mean = None
        self.std = None
        self.loss = []

    def find_params(self, X):
        self.mean = np.array([np.mean(line) for line in X.T])
        self.std = np.array([np.std(line) for line in X.T])

    def standardize(self, X):
        res = []
        for i in range(X.T.shape[0]):
            if i >= 7:
                res.append(X.T[i])
            else:
                res.append((X.T[i] - self.mean[i]) / self.std[i])
        return np.array(res).T

    def reg(self):
        if self.reg_type == "Ridge":
            return 2 * self.eta * self.lambda_ * self.w
        elif self.reg_type == "Lasso":
            return self.eta * self.lambda_ * np.sign(self.w)
        else:
            return 0

    def fit(self, X, y, epoch=1):
        self.epoch = epoch
        self.find_params(X)
        self.X = self.standardize(X)
        self.y = y
        for e in range(epoch):
            for i in range(self.X.shape[0]):
                y_pred = np.dot(self.w, self.X[i]) + self.b
                self.w = self.w - 2 * self.eta * (y_pred - self.y[i]) * self.X[i] - self.reg()
                self.b = self.b - 2 * self.eta * (y_pred - self.y[i])
                self.loss.append(compute_error(y_pred, y[i]))
        return self.w, self.b

    def predict(self, X):
        X_standardized = self.standardize(X)
        return np.dot(X_standardized, self.w.T) + self.b

    def plot_loss(self, path):
        plt.figure()
        steps = [i + 1 for i in range(len(self.loss))]
        plt.plot(steps, self.loss)
        plt.xlabel("Number of steps")
        plt.ylabel("Training loss")
        if self.reg_type == "Ridge":
            reg_type = "$L_2$"
        elif self.reg_type == "Lasso":
            reg_type = "$L_1$"
        else:
            reg_type = "no"
        title = "Training loss: {} epochs, {} regularization, $\eta$ = {}".format(self.epoch, reg_type, self.eta)
        if self.reg_type is not None:
            title += ", $\lambda$ = {}".format(self.lambda_)
        plt.title(title)
        plt.savefig(path)


if __name__ == "__main__":
    X_train, y_train = read_data("carseats_train.csv")
    X_test, y_test = read_data("carseats_test.csv")

    LR = LinearRegression(12, 0.01)
    LR.fit(X_train, y_train, epoch=50)
    # print(LR.w)
    print(compute_error(LR.predict(X_test), y_test))
    LR.plot_loss("result/1.png")

    LR = LinearRegression(12, 0.001)
    LR.fit(X_train, y_train, epoch=50)
    # print(LR.w)
    print(compute_error(LR.predict(X_test), y_test))
    LR.plot_loss("result/2.png")

    LR = LinearRegression(12, 0.001, regularization="Ridge", penalty=0.1)
    LR.fit(X_train, y_train, epoch=50)
    # print(LR.w)
    print(compute_error(LR.predict(X_test), y_test))
    LR.plot_loss("result/3.png")

    LR = LinearRegression(12, 0.001, regularization="Lasso", penalty=0.1)
    LR.fit(X_train, y_train, epoch=50)
    # print(LR.w)
    print(compute_error(LR.predict(X_test), y_test))
    LR.plot_loss("result/4.png")
