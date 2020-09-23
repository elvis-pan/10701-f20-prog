import math

_EPSILON = 1e-9  # small value to ensure the variance is not 0


def read_attribute(filename):
    dictionaries = []
    discrete = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            if "continuous" not in line:
                a = line.strip("\n").strip(".").split(", ")
                d = dict()
                for i in range(len(a)):
                    d[a[i]] = i
                dictionaries.append(d)
                discrete.append(True)
            else:
                dictionaries.append(None)
                discrete.append(False)
        file.close()
    return dictionaries, discrete


def transform_data(data, dictionaries, discrete):
    for line in data:
        for i in range(len(line)):
            if discrete[i]:
                line[i] = dictionaries[i][line[i]]
            else:
                try:
                    line[i] = int(line[i])
                except:
                    pass
    return data


def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            if "?" not in line:  # incomplete data
                data.append(line.strip("\n").strip(".").split(", "))
        file.close()
    return data


def list_mean(x):
    return sum(x) / len(x)


def list_variance(x, mean):
    return sum([(y - mean) ** 2 for y in x]) / len(x)


def list_std(x, mean):
    return math.sqrt(list_variance(x, mean))


def gaussian_pdf(x, mean, std):
    e = math.exp(-(math.pow(x - mean, 2)) / (2 * (math.pow(std) + _EPSILON)))
    return (1 / (math.sqrt(2 * math.pi * (pow(std) + _EPSILON)))) * e


class NaiveBayes(object):
    def __init__(self, dictionaries, discrete):
        self.dictionaries = dictionaries
        self.discrete = discrete
        self.params = [[], []]  # self.params[c][i][j]
        self.prior = [0, 0]
        for i in range(len(dictionaries) - 1):
            if self.discrete[i]:  # discrete attribute, categorical distribution
                theta_0 = [0 for i in range(len(dictionaries[i]))]
                theta_1 = [0 for i in range(len(dictionaries[i]))]
                self.params[0].append(theta_0)
                self.params[1].append(theta_1)
            else:  # continuous attribute, mean and std
                self.params[0].append((0, 0))
                self.params[1].append((0, 0))

    def fit(self, data):
        data_cat = [[line for line in data if line[-1] == 0],
                    [line for line in data if line[-1] == 1]]
        self.prior = [len(data_cat[0]) / len(data), len(data_cat[1]) / len(data)]
        for cat in [0, 1]:
            for i in range(len(dictionaries) - 1):
                values = [line[i] for line in data_cat[cat]]
                if self.discrete[i]:
                    length = len(data_cat[cat])
                    for (key, value) in self.dictionaries[i].items():
                        num_valid = len([line for line in data_cat[cat] if line[i] == value])
                        self.params[cat][i][value] = num_valid / length
                else:
                    mean = list_mean(values)
                    std = list_std(values, mean)
                    self.params[cat][i] = (mean, std)

    def p(self, cat, attribute_id, value):
        if discrete[attribute_id]:
            return self.params[cat][attribute_id][dictionaries[attribute_id][value]]
        else:
            (mean, std) = self.params[cat][attribute_id]
            return gaussian_pdf(value, mean, std)

    def get_log_posterior(self, line):
        cat = line[-1]
        res = math.log(self.prior[cat])
        for i in range(len(line) - 1):
            res += math.log(self.p(cat, i, line[i]))
        return res

    def predict(self, data):
        for line in data:
            for c in [0, 1]:
                pass
        pass


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
