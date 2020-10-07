# 10-701 Homework 2 Programming
Author: Elvis Pan

Email: ypan2@andrew.cmu.edu

## Files

| File                | Description      |
|---------------------|------------------|
|`LinearRegression.py`| Code             |
|`carseats_train.csv` | Training dataset |
|`carseats_test.csv`  | Testing dataset  |

## Run
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