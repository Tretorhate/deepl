import numpy as np
from nn_core import SimpleNet, ReLU, Sigmoid


def make_data():
    X = np.array([
        [0.2, 0.8, 0.7, 0.1],
        [0.1, 0.7, 0.6, 0.0],
        [0.3, 0.9, 0.8, 0.2],
        [0.8, 0.3, 0.4, 0.9],
        [0.9, 0.2, 0.3, 1.0],
        [0.7, 0.4, 0.5, 0.8],
        [0.15, 0.85, 0.65, 0.05],
        [0.85, 0.25, 0.35, 0.95],
        [0.25, 0.75, 0.9, 0.3],
        [0.75, 0.35, 0.45, 0.7],
        [0.05, 0.95, 0.55, 0.15],
        [0.95, 0.05, 0.25, 0.85]
    ]).T
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1])
    return X, y


def test(net, X, y, label):
    c = 0
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        t = y[i]
        p = net.predict(x)
        c += (p == t)
    a = c / len(y) * 100
    print(f"{label} acc: {a:.1f}% ({c}/{len(y)})")


if __name__ == "__main__":
    # 4 -> 8 -> 4 -> 2 network, compare ReLU vs Sigmoid on same data
    net_relu = SimpleNet([4, 8, 4, 2], ReLU())
    net_sig = SimpleNet([4, 8, 4, 2], Sigmoid())

    X, y = make_data()
    test(net_relu, X, y, "ReLU")
    test(net_sig, X, y, "Sig")