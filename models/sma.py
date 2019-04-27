import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 10
samples = 1000


def SMA(x, N):
    window = []
    out = []
    h = x[0]
    for v in x:
        if len(window) == N:
            window.pop(0)
        window.append(v)
        out.append(torch.cat(window).mean().view(1, 1))
    return torch.cat(out)


class SMA_RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1, 1, 1)
    
    def forward(self, x):
        x, _ = self.gru(x, x[0].repeat(1, 1, 1))
        return x


def test():
    x = torch.randn(samples, 1)
    y = SMA(x, N)
    
    sma_rnn = SMA_RNN()
    optimizer = torch.optim.Adam(sma_rnn.parameters(), lr=0.1)
    
    for e in range(100):
        optimizer.zero_grad()
        y_pred = sma_rnn(x.view(samples, 1, -1))
        loss = ((y - y_pred) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        print(f'\r{e}: {loss.detach().numpy()}', end='')
    
    plt.subplot(2, 1, 1)
    plt.plot(x.numpy())
    plt.plot(y.numpy())
    plt.subplot(2, 1, 2)
    plt.plot(y_pred.detach().numpy().squeeze())
    plt.plot(y.numpy())
    plt.savefig('images/sma_rnn_compare.png')
