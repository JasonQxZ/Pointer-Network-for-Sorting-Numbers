# -*- coding: UTF-8 -*-
import numpy as np
import torch

from model import Pointer_network

MAX_EPOCH = 100000
batch_size = 256
shiyan_setting = {1: 5, 2: 10, 3: 5}
shiyan = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Pointer_network(input_dim=1,
                        hidden_dim=shiyan_setting[shiyan],
                        device=device).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
def loss_fun(yhat, y):
    yonehot = torch.zeros(yhat.shape).scatter_(2, y.unsqueeze(2), 1)
    return -torch.mean(yonehot*torch.log(yhat))


def getdata(shiyan=1, batch_size=batch_size):
    if shiyan == 1:
        high = 100
        senlen = 5
        x = np.array([np.random.choice(range(high), senlen, replace=False)
                      for _ in range(batch_size)])
        y = np.argsort(x)
    elif shiyan == 2:
        high = 100
        senlen = 10
        x = np.array([np.random.choice(range(high), senlen, replace=False)
                      for _ in range(batch_size)])
        y = np.argsort(x)
    elif shiyan == 3:
        senlen = 5
        x = np.array([np.random.random(senlen) for _ in range(batch_size)])
        y = np.argsort(x)
    return x, y


def evaluate():
    accuracy_sum = 0.0
    for _ in range(300):
        test_x, test_y = getdata(shiyan=shiyan)
        prediction = model(test_x)
        accuracy = np.mean(torch.argmax(
            prediction, dim=2).cpu().numpy() == test_y)
        accuracy_sum += accuracy
    print('accuracy is ', accuracy_sum/(300.0))


print('Epoch:\tLoss:')
for epoch in range(1, MAX_EPOCH):
    train_x, train_y = getdata(shiyan=shiyan)
    prediction = model(train_x)
    loss = loss_fun(prediction.cpu(), torch.from_numpy(train_y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 300 == 0:
        print('{}\t{:.4f}'.format(epoch, loss.item()))
    if epoch % 2000 == 0:
        evaluate()
