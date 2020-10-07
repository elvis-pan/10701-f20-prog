# 10-701 Fall 2020 Programming Homework
Programming Homework of CMU's 10-701 Introduction to Machine Learning (PhD).

All solutions by Elvis Yan Pan (CMU 23')

## Homework 1
A Na&iuml;ve Bayes classifier with both discrete and continuous attributes.

Discrete attributes are modeled by categorical distribution with parameter <img src="https://latex.codecogs.com/gif.latex?\alpha_{i,c,j}">, which shows the probability of the ith attribute being j given label c.

Continuous attributes are modeled by Gaussian distribution with parameter <img src="https://latex.codecogs.com/gif.latex?\mu"> and <img src="https://latex.codecogs.com/gif.latex?\sigma^2">, where the probability is computed using
<img src="https://latex.codecogs.com/gif.latex?\mathrm{P}(x)=\frac{1}{\sqrt{2\pi(\sigma^2 %2B \epsilon)}}\exp\left(-\frac{(x-\mu)^2}{2(\sigma^2 %2B \epsilon)}\right)">, where <img src="https://latex.codecogs.com/gif.latex?\epsilon = 10^{-9}"> is a small factor to ensure the variance is not 0.

Run: `cd hw1` and then `python3 NaiveBayes.py`.

## Homework 2
Linear Regression with stochastic gradient descent and L2, L1 regularization (ridge regression and lasso regression).

In stochastic gradient descent, the gradient of the loss function is replaced by an estimate from a single datapoint, mathematically <img src="https://latex.codecogs.com/gif.latex?\mathcal{L} \approx (\hat{y}^{(i)} - y^{(i)})^2">.

The update rules are <img src="https://latex.codecogs.com/gif.latex?w_j \leftarrow w_j - 2\eta(\hat{y}^{(i)} - y^{(i)})x_j^{(i)}">

### Files

| File                | Description      |
|---------------------|------------------|
|`LinearRegression.py`| Code             |
|`carseats_train.csv` | Training dataset |
|`carseats_test.csv`  | Testing dataset  |

### Run
To reproduce the four configurations specified in the problem, run `$ python3 LinearRegression.py` in the directory, and the program will generate the plots in the `result` folder and print the testing loss of the four training configurations.

To try other configurations, import the file `LinearRegression.py` and run it like the following example.
```python
# m : the number of features
# e : the learning rate
# r : the regularization, None for no, "Ridge" for ridge, and "Lasso" for lasso
# l : the tuning parameter

X, y = read_data("path/to/data")
LR = LinearRegression(m, e, regularization=r, penalty=l)
w, b = LR.fit(X, y)
LR.plot_loss("name/of/output")
y_pred = LR.predict(X)
```