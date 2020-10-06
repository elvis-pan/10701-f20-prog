import numpy as np

class LinearRegression:
    def __init__(self, num_feature, learn_rate, reg_type=None, reg=0):
        self.m = num_feature
        self.eta = learn_rate
        self.w = np.zeros(num_feature)
        self.b = 0
        self.reg_type = reg_type
        self.lambda_ = reg

    def reg(self):
        if (self.reg_type == None):
            return 0
        if (self.reg_type == "Ridge"):
            return self.lambda_ * np.pow(self.w, 2)
        if (self.reg_type == "Lasso"):
            return self.lambda_ * np.abs(self.w)

    def fit(self, X, y, epoch=1):
        for e in range(epoch):
            for i in range(X.shape[0]):
                y_pred = np.dot(self.w, X[i]) + self.b 
                self.w -= 2 * self.eta * (y_pred - y) * X[i] + self.reg()
                self.b -= 2 * self.eta * (y_pred - y)

    def predict(self, X):
        return np.dot(X, self.w.T) + self.b


if __name__ == "__main__":
    pass
