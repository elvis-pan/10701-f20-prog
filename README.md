# 10-701 Fall 2020 Programming Homework
Programming Homework of CMU's 10-701 Introduction to Machine Learning (PhD).

Author: Elvis Pan

Email: ypan2@andrew.cmu.edu

## Homework 1
A Na&iuml;ve Bayes classifier with both discrete and continuous attributes.

Discrete attributes are modeled by categorical distribution with the proportion of each category of the attribute in the dataset. Continuous attributes are modeled by Gaussian distribution with the mean and variance of the attribute in the dataset.

### Files
| File           | Description               |
|----------------|---------------------------|
|`NaiveBayes.py` | Code                      |
|`adult.data`    | Training dataset          |
|`adult.test`    | Testing dataset           |
|`attribute.data`| Description of attributes |

### Run
To reproduce the configurations specified in the problem, `% cd hw1` and then run `% python3 NaiveBayes.py`. The results will be printed out.

To try other configurations, import the file `NaiveBayes.py` and run it like the following example.
```python
dicts, disc = read_attribute("/path/to/attribute")
data = read_data("/path/to/data")

NB = NaiveBayes(dicts, disc)
NB.fit(data)
NB.predict(data)
print(NB.accuracy)
```

## Homework 2
Linear Regression with stochastic gradient descent and support for L2, L1 regularization (ridge regression and lasso regression).

In stochastic gradient descent, the gradient of the loss function is replaced by an estimate from a single datapoint.

### Files

| File                | Description      |
|---------------------|------------------|
|`LinearRegression.py`| Code             |
|`carseats_train.csv` | Training dataset |
|`carseats_test.csv`  | Testing dataset  |

### Run
To reproduce the four configurations specified in the problem, `% cd hw2` and run `% python3 LinearRegression.py`, and the program will generate the plots in the `result` folder and print the testing loss of the four training configurations.

To try other configurations, import the file `LinearRegression.py` and run it like the following example.
```python
# m : the number of features
# e : the learning rate
# r : the regularization, None for no, "Ridge" for ridge, and "Lasso" for lasso
# l : the tuning parameter

X, y = read_data("path/to/data")
LR = LinearRegression(m, e, regularization=r, penalty=l)
w, b = LR.fit(X, y, epoch=50)
LR.plot_loss("name/of/output")
y_pred = LR.predict(X)
print(compute_loss(y_pred, y))
```

## Homework 3
Author: Elvis Pan

Email: ypan2@andrew.cmu.edu

### Files

| File            | Description      |
|-----------------|------------------|
|`ANN.py`         | Code             |
|`data/train.csv` | Training dataset |
|`data/test.csv`  | Testing dataset  |

### Run
To reproduce the four configurations specified in the problem, run `$ python3 ANN.py` in the directory, and the program will generate the plots in the `result` folder and print the testing loss of the four training configurations.

To try other configurations, import the file `ANN.py` and run it like the following example.
```python
train_x, train_y = read_data("/path/to/data")
Model = ANN(M, D, K, learning_rate=0.01)
Model.fit(train_x, train_y, epoch=100)
print(Model.compare(Model.predict(train_x), train_y))
```
## Homework 4 
Author: Elvis Pan

Email: ypan2@andrew.cmu.edu

### Files

| File                         | Description             |
|------------------------------|-------------------------|
|`AdaBoost.py`                 | Code for AdaBoost       |
|`DecisionStump.py`            | Code for decision stump |
|`datasets/train_adaboost.csv` | Training dataset        |
|`datasets/test_adaboost.csv`  | Testing dataset         |

### Run
To reproduce the solution to the questions, run `$ python3 AdaBoost.py` in the directory, and the program will generate the plots in the `result` folder and print the testing loss of the four training configurations.

To try other configurations, import the file `AdaBoost.py` and run it like the following example.
```python
train_x, train_y = read_data("/path/to/data")
AB = AdaBoost()
AB.fit(train_x, train_y, iteration=50)
print(AB.eval_model(train_y, AB.predict(train_x)))
```
