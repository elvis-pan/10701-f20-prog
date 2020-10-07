import csv
import numpy as np
import matplotlib.pyplot as plt


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
    return data


def split_data(data):
    X = np.array([line[1:] for line in data])
    y = np.array([line[0] for line in data])
    return X, y


def compute_error(y_pred, y):
    return np.mean(np.power(y_pred - y, 2))


class LinearRegression:
    def __init__(self, num_feature, learn_rate, reg_type=None, reg=0):
        self.m = num_feature
        self.eta = learn_rate
        self.w = np.zeros(num_feature, dtype=np.float64)
        self.b = 0.0
        self.reg_type = reg_type if reg_type is not None else "no"
        self.lambda_ = reg
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
        if self.reg_type == "no":
            return 0
        elif self.reg_type == "ridge":
            return 2 * self.eta * self.lambda_ * self.w
        elif self.reg_type == "lasso":
            return self.eta * self.lambda_ * np.sign(self.w)

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
        x = [i + 1 for i in range(len(self.loss))]
        plt.plot(x, self.loss)
        plt.xlabel("Number of steps")
        plt.ylabel("Training loss")
        plt.title("Error of $\eta$ = {}, {} epochs, {} regularization".format(self.eta, self.epoch, self.reg_type))
        plt.savefig(path)


if __name__ == "__main__":
    train_data = read_data("carseats_train.csv")
    test_data = read_data("carseats_test.csv")
    X_train, y_train = split_data(train_data)
    X_test, y_test = split_data(test_data)

    LR = LinearRegression(12, 0.01)
    LR.fit(X_train, y_train, epoch=50)
    LR.plot_loss("result/1.png")

    LR = LinearRegression(12, 0.001)
    LR.fit(X_train, y_train, epoch=50)
    LR.plot_loss("result/2.png")

    LR = LinearRegression(12, 0.001, reg_type="ridge", reg=0.1)
    LR.fit(X_train, y_train, epoch=50)
    LR.plot_loss("result/3.png")

    LR = LinearRegression(12, 0.001, reg_type="lasso", reg=0.1)
    LR.fit(X_train, y_train, epoch=50)
    LR.plot_loss("result/4.png")
