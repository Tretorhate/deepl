import numpy as np


class FullyConnected:
    def __init__(self, n_in, n_out):
        # small random weights and zero bias
        self.W = np.random.randn(n_out, n_in) * 0.1
        self.b = np.zeros((n_out, 1))

    def forward(self, x):
        return self.W @ x + self.b


class ReLU:
    def forward(self, x):
        return np.maximum(0, x)


class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class SimpleNet:
    """
    Simple feedforward neural network with fully connected layers and
    a single activation function applied to all hidden layers.

    sizes: list/tuple of layer sizes, e.g. [2, 4, 2] or [4, 8, 4, 2]
    activation: object with .forward(x), e.g. ReLU() or Sigmoid()
    """

    def __init__(self, sizes, activation):
        self.layers = [
            FullyConnected(sizes[i], sizes[i + 1])
            for i in range(len(sizes) - 1)
        ]
        self.act = activation

    def forward(self, x):
        h = x
        # apply activation on all but the last (output) layer
        for fc in self.layers[:-1]:
            h = self.act.forward(fc.forward(h))
        return self.layers[-1].forward(h)

    def predict(self, x):
        return np.argmax(self.forward(x))


