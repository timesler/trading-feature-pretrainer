import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 30


def EMA(x, N):
    alpha = 2 / (N + 1)
    out = []
    h = x[0]
    for v in x:
        h = alpha * v + (1 - alpha) * h
        out.append(h)
    return torch.cat(out)


def SMMA(x, N):
    return EMA(x, 2 * N - 1)


class EMA_RNN(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.alpha = 2 / (N + 1)
        self.lin = nn.Linear(2, 1, bias=False)
        self.lin.weight = nn.Parameter(
            torch.Tensor([1 - self.alpha, self.alpha]).view(1, -1),
            requires_grad=False
        )
    
    def forward(self, x):
        out = []
        h = x[0]
        for v in x:
            h = self.lin(torch.cat([h, v]))
            out.append(h)
        return torch.cat(out)


class SMMA_RNN(EMA_RNN):
    def __init__(self, N):
        super().__init__(2 * N - 1)

    
def test():
    x = torch.randn(1000, 1)
    y = EMA(x, N)

    ema_rnn = EMA_RNN(N)
    y_pred = ema_rnn(x)

    print(f'Loss: {torch.mean((y - y_pred) ** 2)}')
    
    plt.subplot(2, 1, 1)
    plt.plot(x.numpy())
    plt.plot(y.numpy())
    plt.subplot(2, 1, 2)
    plt.plot(y_pred.numpy())
    plt.plot(y.numpy())
    plt.savefig('images/ema_rnn_compare.png')

    
if __name__ == "__main__":
    test()
