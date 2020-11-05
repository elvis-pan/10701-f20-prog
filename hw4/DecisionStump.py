import numpy as np


class DecisionStump:
    """
    1-level decision tree, weak classifier for the adaboost algorithm.
    """

    def __init__(self):
        self.f = None
        self.accuracy = 0

    def fit(self, train_x, train_y):
        h = [(x[0], y) for (x, y) in zip(train_x, train_y)]
        v = [(x[1], y) for (x, y) in zip(train_x, train_y)]
        # Compute horizontal boundary
        lowest_err = len(train_y)
        for i in range(len(h)):
            f1 = lambda x: (x[0] < bound and x[1] > 0) or (x[0] > bound and x[1] < 0)
            f2 = lambda x: (x[0] < bound and x[1] < 0) or (x[0] > bound and x[1] > 0)
            # Horizontal
            bound = h[i][0]
            err1 = len(list(filter(f1, h)))
            err2 = len(list(filter(f2, h)))
            if err1 < lowest_err:
                lowest_err = err1
                self.f = lambda x: -1 if x[0] < bound else 1
            if err2 < lowest_err:
                lowest_err = err2
                self.f = lambda x: 1 if x[0] < bound else -1
            # Vertical
            bound = v[i][0]
            err1 = len(list(filter(f1, v)))
            err2 = len(list(filter(f2, v)))
            if err1 < lowest_err:
                lowest_err = err1
                self.f = lambda x: -1 if x[1] < bound else 1
            if err2 < lowest_err:
                lowest_err = err2
                self.f = lambda x: 1 if x[1] < bound else -1
        self.accuracy = lowest_err / len(train_y)
        return self.f

    def predict(test_x, test_y):
        pass
