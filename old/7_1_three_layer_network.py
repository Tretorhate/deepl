import numpy as np
from nn_core import SimpleNet, ReLU


if __name__ == "__main__":
    # 2 -> 8 -> 4 -> 2 network with ReLU on hidden layers
    net = SimpleNet([2, 8, 4, 2], ReLU())
    x = np.array([[0.8], [0.6]])
    out = net.forward(x)
    p = net.predict(x)
    print("Input:", x.T)
    print("Scores:", out.T)
    print("Pred:", p)