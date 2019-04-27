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


def EMSD(x, N):
    x_ema = EMA(x, N)
    alpha = 2 / (N + 1)
    out = []
    h = 0
    for v, v_ema in zip(x, x_ema):
        h = (alpha - alpha **2) * (v - v_ema) ** 2 + (1 - alpha) * h
        out.append(torch.sqrt(h))
    return torch.cat(out)


def SMMA(x, N):
    return EMA(x, 2 * N - 1)


class EMA_RNN(nn.Module):
    def __init__(self, N, trainable=False):
        super().__init__()
        self.N = N
        self.alpha = 2 / (N + 1)
        self.lin = nn.Linear(2, 1, bias=False)
        self.lin.weight = nn.Parameter(
            torch.Tensor([1 - self.alpha, self.alpha]).view(1, -1),
            requires_grad=trainable
        )
    
    def forward(self, x):
        out = []
        h = x[0]
        for v in x:
            h = self.lin(torch.cat([h, v]))
            out.append(h)
        return torch.cat(out)


class EMSD_RNN(nn.Module):
    def __init__(self, N, trainable=False):
        super().__init__()
        self.N = N
        self.alpha = 2 / (N + 1)
        self.lin_ema = nn.Linear(2, 1, bias=False)
        self.lin_ema.weight = nn.Parameter(
            torch.Tensor([1 - self.alpha, self.alpha]).view(1, -1),
            requires_grad=trainable
        )
        self.lin_emv = nn.Linear(2, 1, bias=False)
        self.lin_emv.weight = nn.Parameter(
            torch.Tensor([1 - self.alpha, self.alpha - self.alpha ** 2]).view(1, -1),
            requires_grad=trainable
        )
    
    def forward(self, x):
        out = []
        h_ema = x[0]
        h_emv = x[0] * 0
        for v in x:
            h_ema = self.lin_ema(torch.cat([h_ema, v]))
            d = (v - h_ema) ** 2
            h_emv = self.lin_emv(torch.cat([h_emv, d]))
            out.append(torch.sqrt(h_emv))
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
