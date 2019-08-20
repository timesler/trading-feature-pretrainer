# Notes on equivalent representations of trading indicators

## Exponential moving averages

Can be represented using a single vanilla recurrent node with a single parameter.

## Moving-average convergence divergence

Can be represented as the difference of two EMA nodes.

## Body wick ratio

Equal to: (open - close) / (high - low + &epsilon;)

The numerator and denominator can be calculated using a normal linear layer, but the ratio is not a standard neural network operation. For calculations such as this, I see the options as:

1. Handcode the calculation for all indicators that need it. This has the advantage of replicating the indicator exactly in the output of the network. It has the disadvantage of not being learnable later on - the network can't learn a more effective version of this feature given the task at hand.
1. Attempt to reproduce the indicator using one or more linear+nonlinear layer combinations, leaning on the "general approximation" property of neural networks.
1. Modify or parameterize the expression so that it can be represented directly using the parameters of a conventional neural network. For instance, we could replace the ratio with a difference.

I believe that, for this project to be worthwhile, we must choose one of the latter two options, ensuring that the network remains as learnable as possible.