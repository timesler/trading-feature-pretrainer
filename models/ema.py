import torch
from torch import nn
from torch import optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def EMA(x, N):
    alpha = 2 / (N + 1)
    out = []
    h = x[0]
    for v in x:
        h = alpha * v + (1 - alpha) * h
        out.append(h.view(1, -1))
    return torch.cat(out, 0)


def EMSD(x, N):
    alpha = 2 / (N + 1)
    out = []
    h_a = x[0]
    h_v = 0
    for v in x:
        h_v = (alpha - alpha **2) * (v - h_a) ** 2 + (1 - alpha) * h_v
        h_a = alpha * v + (1 - alpha) * h_a
        out.append(torch.sqrt(h_v).view(1, -1))
    return torch.cat(out, 0)


def SMMA(x, N):
    return EMA(x, 2 * N - 1)


def SMMSD(x, N):
    return EMSD(x, 2 * N - 1)


class EMA_layer(nn.Module):
    def __init__(self, input_dim, num_N, init_N: int=None, with_SD=False, trainable=True):
        super().__init__()

        self.input_dim = input_dim
        self.num_N = num_N
        self.with_SD = with_SD

        if init_N is not None:
            if not isinstance(init_N, list):
                alpha_p = torch.ones(num_N, input_dim) * 2 / (init_N + 1)
            elif len(init_N) == num_N:
                alpha_p = 2 / (torch.FloatTensor(init_N).view(num_N, -1).expand(-1, input_dim) + 1)
            else:
                raise Exception("`init_N` should be either a scalar or a list of length `num_N`.")
        else:
            alpha_p = torch.FloatTensor(num_N, input_dim).uniform_(0, 1)

        self.alpha = nn.Parameter(alpha_p, requires_grad=trainable)
    
    def forward(self, x, h):
        if self.with_SD:
            h_a, h_s = h
            h_v = h_s ** 2
        else:
            h_a = h
        out_a = []
        out_s = []
        for x_i in x:
            d = x_i - h_a
            if self.with_SD:
                h_v = (1 - self.alpha) * (h_v + self.alpha * d**2)
                out_s.append(h_v.sqrt())
            h_a = self.alpha * d + h_a
            out_a.append(h_a)
        out_a = torch.cat(out_a).view(x.shape[0], self.num_N, self.input_dim)
        if self.with_SD:
            out_s = torch.cat(out_s).view(x.shape[0], self.num_N, self.input_dim)
            out = torch.cat([out_a, out_s], 1)
            hN = (h_a, h_v.sqrt())
        else:
            out, hN = out_a, h_a
        return out, hN
    
    def get_h0(self, x_0):
        h0 = x_0.expand(self.num_N, self.input_dim)
        if self.with_SD:
            h0 = (h0, h0 * 1e-8)
        return h0


class SMMA_layer(EMA_layer):
    def __init__(self, input_dim, num_N, init_N: int=None, with_SD=False, trainable=True):
        if init_N is None:
            init_N = None
        elif isinstance(init_N, list):
            init_N = [2 * N - 1 for N in init_N]
        else:
            init_N = 2 * init_N - 1
        super().__init__(input_dim, num_N, init_N, with_SD, trainable)

    
def test():
    samples = 1000
    inputs = 2
    N = [5, 50]

    print('Testing EMA RNN')
    x = torch.randn(samples, inputs)
    y = [EMA(x, N_i) for N_i in N] + [EMSD(x, N_i) for N_i in N]
    y = torch.cat(y, 1).view(samples, -1, inputs)

    print('With pre-calculated weights (per-sample):')
    ema_rnn = EMA_layer(inputs, len(N), N, with_SD=True)
    hN = ema_rnn.get_h0(x[0])
    
    y_pred = []
    for x_i in x:
        y_pred_i, hN = ema_rnn(x_i.view(1, -1), hN)
        y_pred.append(y_pred_i)
    y_pred = torch.cat(y_pred, 0)
    print(f'  Loss: {torch.mean((y - y_pred) ** 2)}')
    
    fig = plt.figure(figsize=(12, 9))
    cnt = 1
    for i in range(inputs):
        ax = fig.add_subplot(2, inputs, cnt)
        ax.plot(y[:, :, i].squeeze().numpy())
        ax = fig.add_subplot(2, inputs, cnt + 1)
        ax.plot(y_pred[:, :, i].detach().squeeze().numpy())
        cnt += 2
    plt.savefig('images/ema_rnn_compare_precalculated.png')

    print('With learned weights:')
    ema_rnn = EMA_layer(inputs, len(N), with_SD=True)
    h0 = ema_rnn.get_h0(x[0])
    optimizer = optim.SGD(ema_rnn.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        y_pred, hN = ema_rnn(x, h0)
        loss = torch.mean((y - y_pred) ** 2)
        loss.backward()
        optimizer.step()
        print(f'\r  {epoch} - Loss: {loss}      ', end='')
    
    print('')
    
    fig = plt.figure(figsize=(12, 9))
    cnt = 1
    for i in range(inputs):
        ax = fig.add_subplot(2, inputs, cnt)
        ax.plot(y[:, :, i].squeeze().numpy())
        ax = fig.add_subplot(2, inputs, cnt + 1)
        ax.plot(y_pred[:, :, i].detach().squeeze().numpy())
        cnt += 2
    plt.savefig('images/ema_rnn_compare_learned.png')

    
if __name__ == "__main__":
    test()
