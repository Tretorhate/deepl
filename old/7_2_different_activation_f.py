import numpy as np
from nn_core import SimpleNet, ReLU, Sigmoid


if __name__ == "__main__":
    x = np.array([[0.8], [0.6]])

    # same architecture 2 -> 4 -> 2 with different activations
    net_sig = SimpleNet([2, 4, 2], Sigmoid())
    out_sig = net_sig.forward(x)
    p_sig = net_sig.predict(x)

    net_relu = SimpleNet([2, 4, 2], ReLU())
    out_relu = net_relu.forward(x)
    p_relu = net_relu.predict(x)

    print("Sig:", out_sig.T, "p", p_sig)
    print("ReLU:", out_relu.T, "p", p_relu)