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
    pass


if __name__ == "__main__":
    train_x, train_y = read_data("datasets/train_adaboost.csv")
    DS = DecisionStump()
    DS.fit(train_x, train_y)
    print(DS.accuracy)
