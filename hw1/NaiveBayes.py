import math
import matplotlib.pyplot as plt

_EPSILON = 1e-9  # small value to ensure the variance is not 0


def read_attribute(filename):  # read the attributes
    dicts = []  # dictionaries to convert discrete values to integers
    disc = []  # disc[i] = whether the ith attribute is discrete
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


def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            if "?" not in line:  # incomplete data
                l = line.strip("\n").strip(".").split(", ")
                if len(l) == 15: # filter out empty data
                    data.append(l)
        file.close()
    return data


def read_data_n(filename, n):
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines();
        for i in range(n):
            if "?" not in lines[i]:  # incomplete data
                l = lines[i].strip("\n").strip(".").split(", ")
                if len(l) == 15: # filter out empty data
                    data.append(l)
        file.close()
    return data


def transform_data(data, dicts, disc):
    for line in data:
        for i in range(len(line)):
            if disc[i]:  # discrete, transform to value in dict
                line[i] = dicts[i][line[i]]
            else:  # continuous, change type to int
                line[i] = int(line[i])
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


def diff(x, y):
    assert len(x) == len(y)
    res = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            res += 1
    return res / len(x)


def plot_result(ns, train_acc, test_acc):
    plt.figure()
    plt.plot(ns, train_acc, label="Train accuracy")
    plt.plot(ns, test_acc, label="Test accuracy")
    plt.xlabel("Exponent of number of data points selected")
    plt.ylabel("Test accuracy")
    plt.legend()
    plt.savefig("result.png")


class NaiveBayes:
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

    def fit(self, data):  # calculate the best fit parameter using data
        data_label = [[line for line in data if line[-1] == 0],
                    [line for line in data if line[-1] == 1]]
        self.prior = [len(data_label[0]) / len(data),
                      len(data_label[1]) / len(data)]
        for label in [0, 1]:
            for i in range(len(self.dicts) - 1):
                values = [line[i] for line in data_label[label]]
                if self.disc[i]:
                    length = len(data_label[label])
                    for (key, value) in self.dicts[i].items():
                        num_valid = len([line for line in data_label[label]
                                         if line[i] == value])
                        self.params[label][i][value] = num_valid / length
                else:
                    mean = list_mean(values)
                    var = list_variance(values, mean)
                    self.params[label][i] = (mean, var)

    def p(self, label, attribute_id, value):  # the probability function
        if self.disc[attribute_id]:  # categorical distribution
            return self.params[label][attribute_id][value]
        else:  # gaussian distribution
            (mean, var) = self.params[label][attribute_id]
            return gaussian_pdf(value, mean, var)

    def log_posterior(self, line, label):
        res = math.log(self.prior[label])
        for i in range(len(line) - 1):
            prob = self.p(label, i, line[i])
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
            result.append(0 if lp_0 > lp_1 else 1)
        self.result = result
        self.accuracy = diff(result, answer)
        return result


if __name__ == "__main__":
    dicts, disc = read_attribute("attribute.data")
    train_data = read_data("adult.data")
    test_data = read_data("adult.test")
    del test_data[0]
    train_data = transform_data(train_data, dicts, disc)
    test_data = transform_data(test_data, dicts, disc)

    NB = NaiveBayes(dicts, disc)  # new Naive Bayes classifier
    NB.fit(train_data)

    print("Prior probability of each class.")  # the prior, answer to 5.1.1
    print("<=50K:", NB.prior[0])
    print(">50K:", NB.prior[1])

    print("\nClass >50K:")  # parameter for class >50K
    print("education-num:", NB.params[1][4])
    print("martial-status:", NB.params[1][5])
    print("race:", NB.params[1][8])
    print("capital-gain:", NB.params[1][10])

    print("\nClass <=50K:")  # parameter for class <=50K
    print("education-num:", NB.params[0][4])
    print("martial-status:", NB.params[0][5])
    print("race:", NB.params[0][8])
    print("capital-gain:", NB.params[0][10])

    # Report the log-posterior values for the first 10 test data
    print("\nLog-posterior values for the first 10 test data")
    print([NB.log_posterior(test_data[i], test_data[i][-1]) for i in range(10)])

    NB.predict(train_data)
    print("Train accuracy: ", NB.accuracy)
    NB.predict(test_data)
    print("\nTest accuracy: ", NB.accuracy)

    train_acc = []
    test_acc = []
    print("\nAccuracy on training n data")
    for i in range(5, 14):
        train_data_n = read_data_n("adult.data", 2 ** i)
        train_data_n = transform_data(train_data_n, dicts, disc)
        NB = NaiveBayes(dicts, disc)
        NB.fit(train_data_n)
        NB.predict(train_data_n)
        print("Train accuracy of n={}:".format(2 ** i), NB.accuracy)
        train_acc.append(NB.accuracy)
        NB.predict(test_data)
        print("Test accuracy of n={}:".format(2 ** i), NB.accuracy)
        test_acc.append(NB.accuracy)
    plot_result(range(5, 14), train_acc, test_acc)
