import math
import matplotlib.pyplot as plt

_EPSILON = 1e-9  # small value to ensure the variance is not 0


def read_attribute(filename):
    dicts = []
    disc = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            if "continuous" not in line:
                a = line.strip("\n").strip(".").split(", ")
                d = dict()
                for i in range(len(a)):
                    d[a[i]] = i
                dicts.append(d)
                disc.append(True)
            else:
                dicts.append(None)
                disc.append(False)
        file.close()
    return dicts, disc


def transform_data(data, dicts, disc):
    for line in data:
        for i in range(len(line)):
            if disc[i]:
                line[i] = dicts[i][line[i]]
            else:
                line[i] = int(line[i])
    return data


def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            if "?" not in line:  # incomplete data
                l = line.strip("\n").strip(".").split(", ")
                if len(l) == 15:
                    data.append(l)
        file.close()
    return data


def list_mean(x):
    return sum(x) / len(x)


def list_variance(x, mean):
    return sum([(y - mean) ** 2 for y in x]) / len(x)


def list_std(x, mean):
    return math.sqrt(list_variance(x, mean))


def gaussian_pdf(x, mean, var):
    e = math.exp(-(math.pow(x - mean, 2)) / (2 * (var + _EPSILON)))
    return (1 / (math.sqrt(2 * math.pi * (var + _EPSILON)))) * e


def compare(x, y):
    assert len(x) == len(y)
    res = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            res += 1
    return res / len(x)


def plot_result(x, y):
    pass


class NaiveBayes(object):
    def __init__(self, dicts, disc):
        self.dicts = dicts
        self.disc = disc
        self.params = [[], []]  # self.params[c][i][j]
        self.prior = [0, 0]
        self.result = []
        self.accuracy = 0
        for i in range(len(dicts) - 1):
            if self.disc[i]:  # discrete attribute, categorical distribution
                theta_0 = [0 for i in range(len(dicts[i]))]
                theta_1 = [0 for i in range(len(dicts[i]))]
                self.params[0].append(theta_0)
                self.params[1].append(theta_1)
            else:  # continuous attribute, mean and variance
                self.params[0].append((0, 0))
                self.params[1].append((0, 0))

    def fit(self, data):
        data_cat = [[line for line in data if line[-1] == 0],
                    [line for line in data if line[-1] == 1]]
        self.prior = [len(data_cat[0]) / len(data), len(data_cat[1]) / len(data)]
        for cat in [0, 1]:
            for i in range(len(dictionaries) - 1):
                values = [line[i] for line in data_cat[cat]]
                if self.disc[i]:
                    length = len(data_cat[cat])
                    for (key, value) in self.dicts[i].items():
                        num_valid = len([line for line in data_cat[cat] if line[i] == value])
                        self.params[cat][i][value] = num_valid / length
                else:
                    mean = list_mean(values)
                    var = list_variance(values, mean)
                    self.params[cat][i] = (mean, var)

    def p(self, cat, attribute_id, value):
        if discrete[attribute_id]:
            return self.params[cat][attribute_id][value]
        else:
            (mean, var) = self.params[cat][attribute_id]
            return gaussian_pdf(value, mean, var)

    def log_posterior(self, line, cat):
        res = math.log(self.prior[cat])
        for i in range(len(line) - 1):
            prob = self.p(cat, i, line[i])
            if prob == 0.0:
                return -float('inf')
            res += math.log(prob)
        return res

    def predict(self, data):
        result = []
        answer = [line[-1] for line in data]
        for line in data:
            lp_0 = self.log_posterior(line, 0)
            lp_1 = self.log_posterior(line, 1)
            result.append(1 if lp_0 < lp_1 else 0)
        self.result = result
        self.accuracy = compare(result, answer)
        return result


if __name__ == "__main__":
    dictionaries, discrete = read_attribute("attribute.data")
    train_data = read_data("adult.data")
    test_data = read_data("adult.test")
    del test_data[0]
    train_data = transform_data(train_data, dictionaries, discrete)
    test_data = transform_data(test_data, dictionaries, discrete)
    NB = NaiveBayes(dictionaries, discrete)
    NB.fit(train_data)
    print(NB.prior)
    NB.predict(test_data)
    print("Accuracy of test data: ", NB.accuracy)
    NB.predict(train_data)
    print("Accuracy of train data: ", NB.accuracy)

    # Report the log-posterior values for the first 10 test data
    # for i in range(10):
    #    print(NB.log_posterior(test_data[i], test_data[i][-1]))

