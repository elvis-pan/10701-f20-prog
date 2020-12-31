import torch
import torch.nn as nn
import torch.optim as optim
from ANN import read_data

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.linear1 = nn.Linear(784, 256)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(256, 10)
        self.softmax = nn.Softmax(10)

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.softmax(self.linear2(x))
        return x

if __name__ == "__main__":
    train_x, train_y = read_data("data/train.csv")
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

    net = ANN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters, lr=0.01)

    for epoch in range(100):
        for x,y in train_x, train_y:
            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    print(loss.values)

