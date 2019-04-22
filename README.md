# trading-feature-pretrainer
This repo contains code for generating pretrained networks equivalent to a wide variety of trading indicators, such as moving averages, MACD, RSI, and many more colorful metrics.

The goal of this project is to generate neural networks that act as function approximators for an arbitrary set of deterministic transformations relevant to forex/stock/derivative market trading. These generated networks should form the input layers to networks built for specific purposes, such as policy gradient reinforcement learning trading agents. Depending on the application, generated network weights should be either fixed, making the generated networks act as generators of known features, or trained further in an application-specific context.

Note that although for some indicators, neural networks are at best function _approximators_ (e.g., simple moving averages), other indicators can be represented _exactly_ using carefully chosen neural network architectures and weights (e.g., exponential moving averages and moving average convergence divergence).

## A simple illustration

Consider an exponential moving average, implemented by the following recursive equation:

&emsp;_EMA<sub>&alpha;</sub>_(_t_) = _&alpha;x_(_t_) + (1 - _&alpha;_)_EMA<sub>&alpha;</sub>_(_t_ - 1)

&emsp;&emsp;where _&alpha;_ = 2/(_N_ + 1) for window _N_.

An exactly equivalent RNN can be constructed using a single recurrent neuron. The neuron accepts a single input, _x_, the hidden state is equivalent to _EMA<sub>&alpha;</sub>_(_t_ - 1), and the output is equivalent to _EMA<sub>&alpha;</sub>_(_t_). From inspection of the equation above, the input and hidden state weights for the neuron are _&alpha;_ and 1 - _&alpha;_, respectively.

<p align="center"><img src="/images/RNN_EMA.png" width="500"/></p>
