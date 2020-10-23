# 10-701 Homework 3 Programming
Author: Elvis Pan

Email: ypan2@andrew.cmu.edu

## Files

| File            | Description      |
|-----------------|------------------|
|`ANN.py`         | Code             |
|`data/train.csv` | Training dataset |
|`data/test.csv`  | Testing dataset  |

## Run
To reproduce the four configurations specified in the problem, run `$ python3 ANN.py` in the directory, and the program will generate the plots in the `result` folder and print the testing loss of the four training configurations.

To try other configurations, import the file `ANN.py` and run it like the following example.
```python
train_x, train_y = read_data("/path/to/data")
Model = ANN(M, D, K, learning_rate=0.01)
Model.fit(train_x, train_y, epoch=100)
print(Model.compare(Model.predict(train_x), train_y))
```