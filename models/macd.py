import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import ema

N_slow = 26
N_fast = 12


def MACD(x, N_slow, N_fast):
    return ema.EMA(x, N_slow) - ema.EMA(x, N_fast)


class MACD_RNN(nn.Module):
    def __init__(self, N_slow, N_fast):
        super().__init__()
        self.N_slow = N_slow
        self.N_fast = N_fast
        self.EMA_slow = ema.EMA_RNN(self.N_slow)
        self.EMA_fast = ema.EMA_RNN(self.N_fast)
    
    def forward(self, x):
        return self.EMA_slow(x) - self.EMA_fast(x)


def test():
    x = torch.randn(1000, 1)
    y = MACD(x, N_slow, N_fast)

    macd_rnn = MACD_RNN(N_slow, N_fast)
    y_pred = macd_rnn(x)

    print(f'Loss: {torch.mean((y - y_pred) ** 2)}')
    
    plt.subplot(2, 1, 1)
    plt.plot(y_pred.numpy())
    plt.subplot(2, 1, 2)
    plt.plot(y.numpy())
    plt.savefig('images/macd_rnn_compare.png')
    

if __name__ == "__main__":
    test()
