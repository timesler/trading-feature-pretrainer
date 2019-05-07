import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt


def EMA(x, N):
    """Function to calculate exponential moving average on timeseries tensors.
    
    Arguments:
        x {torch.FloatTensor} -- 1- or 2-dimensional tensor where the 0th dimension is time.
        N {int} -- Moving average window size.
    
    Returns:
        torch.FloatTensor -- Tensor of EMAs the same shape as x.
    """
    alpha = 2 / (N + 1)
    out = []
    h = x[0]
    for v in x:
        h = alpha * v + (1 - alpha) * h
        out.append(h.view(1, -1))
    return torch.cat(out, 0)


def EMSD(x, N):
    """Function to calculate exponential moving standard deviation on timeseries tensors.
    
    Arguments:
        x {torch.FloatTensor} -- 1- or 2-dimensional tensor where the 0th dimension is time.
        N {int} -- Moving average window size.
    
    Returns:
        torch.FloatTensor -- Tensor of EMSDs the same shape as x.
    """
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
    """Function to calculate smoothed moving average on timeseries tensors."""
    return EMA(x, 2 * N - 1)


def SMMSD(x, N):
    """Function to calculate smoothed moving standard deviation on timeseries tensors."""
    return EMSD(x, 2 * N - 1)


class EMA_layer(nn.Module):
    """Neural network layer that calculates exponential moving averages and standard deviations
    of its inputs.
    """
    def __init__(self, input_dim, num_N, init_N: int=None, with_SD=False, trainable=True):
        """Constructor for EMA_layer module.
        
        Arguments:
            input_dim {int} -- Number of input features.
            num_N {int} -- Number of moving windows/filters.
        
        Keyword Arguments:
            init_N {int} -- Predetermined starting window sizes. (default: {None})
            with_SD {bool} -- Whether or not to also calculate standard deviations for each
                average. (default: {False})
            trainable {bool} -- Whether to make layer parameters trainable. (default: {True})
        """
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
            h_a, h_v = h[0], h[1] **2
        else:
            h_a = h

        out_a = []
        out_v = []
        for x_i in x:
            d = x_i - h_a
            if self.with_SD:
                h_v = (1 - self.alpha) * (h_v + self.alpha * d**2)
                out_v.append(h_v)
            h_a = h_a + self.alpha * d
            out_a.append(h_a)

        out_a = torch.cat(out_a).view(x.shape[0], self.num_N, self.input_dim)
        if self.with_SD:
            out_v = torch.cat(out_v).view(x.shape[0], self.num_N, self.input_dim)
            out, hN = torch.cat([out_a, out_v.sqrt()], 1), (h_a, h_v.sqrt())
        else:
            out, hN = out_a, h_a

        return out, hN
    
    def get_h0(self, x_0):
        """Get appropraite values for the initial hidden state.
        
        Arguments:
            x_0 {torch.FloatTensor} -- The first input sample.
        
        Returns:
            torch.FloatTensor -- Either a single tensor or, if standard deviations are also being
                calculated, a tuple.
        """
        h0 = x_0.expand(self.num_N, self.input_dim)
        if self.with_SD:
            h0 = (h0, torch.ones_like(h0) * 1e-8)
        return h0


class SMMA_layer(EMA_layer):
    """Wrapper of EMA_layer that implements the smoothed moving average."""
    def __init__(self, input_dim, num_N, init_N: int=None, with_SD=False, trainable=True):
        if init_N is None:
            init_N = None
        elif isinstance(init_N, list):
            init_N = [2 * N - 1 for N in init_N]
        else:
            init_N = 2 * init_N - 1
        super().__init__(input_dim, num_N, init_N, with_SD, trainable)

    
def test():
    import cProfile

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
    optimizer = optim.SGD(ema_rnn.parameters(), lr=0.1)

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
