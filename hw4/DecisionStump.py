import numpy as np


class DecisionStump:
    """
    1-level decision tree, weak classifier for the adaboost algorithm.
    """

    def __init__(self):
        self.f = None
        self.direction = None
        self.boundary = 0
        self.y_pred = None
        self.accuracy = 0

    def fit(self, train_x, train_y):
        h = [(x[0], y) for (x, y) in zip(train_x, train_y)]
        v = [(x[1], y) for (x, y) in zip(train_x, train_y)]
        # Compute horizontal boundary
        lowest_err = len(train_y)
        for i in range(len(h) + 1):
            f1 = lambda x: (x[0] < bound and x[1] > 0) or (x[0] >= bound and x[1] < 0)
            f2 = lambda x: (x[0] < bound and x[1] < 0) or (x[0] >= bound and x[1] > 0)
            # Horizontal
            bound = h[i][0] if i < len(h) else float("inf")
            err1 = len(list(filter(f1, h)))
            err2 = len(list(filter(f2, h)))
            if err1 < lowest_err:
                lowest_err = err1
                self.boundary = bound
                self.direction = "horizontal -1"  # horizontal axis < bound is classified as -1
                self.f = lambda x: -1 if x[0] < bound else 1
            if err2 < lowest_err:
                lowest_err = err2
                self.boundary = bound
                self.direction = "horizontal 1"
                self.f = lambda x: 1 if x[0] < bound else -1
            # Vertical
            bound = v[i][0] if i < len(v) else float("inf")
            err1 = len(list(filter(f1, v)))
            err2 = len(list(filter(f2, v)))
            if err1 < lowest_err:
                lowest_err = err1
                self.boundary = bound
                self.direction = "vertical -1"
                self.f = lambda x: -1 if x[1] < bound else 1
            if err2 < lowest_err:
                lowest_err = err2
                self.direction = "vertical 1"
                self.boundary = bound
                self.f = lambda x: 1 if x[1] < bound else -1
        self.accuracy = lowest_err / len(train_y)
        return self.f

    def predict(self, test_x):
        self.y_pred = np.array(list(map(self.f, test_x)))
        return self.y_pred

    def compute_error(self, y, y_pred):
        self.accuracy = np.sum(np.abs(y + y_pred)) / (2 * y.shape[0])
        return self.accuracy
