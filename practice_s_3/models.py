import torch
import torch.nn as nn
import numpy as np
# --- Part 1: Regularization Models ---
class RegularizedMLP(nn.Module):
    def __init__(self, input_size=784, dropout_rate=0.0):
        super(RegularizedMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

# --- Part 2: RNN Models ---
class VanillaRNNScratch:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Xavier-style initialization
        self.Wxh = np.random.randn(hidden_dim, input_dim) * 0.01
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.Why = np.random.randn(output_dim, hidden_dim) * 0.01
        self.bh = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))

        # Gradients
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dWhy = np.zeros_like(self.Why)
        self.dbh = np.zeros_like(self.bh)
        self.dby = np.zeros_like(self.by)

    def forward(self, inputs):
        """Forward pass through time steps"""
        h = np.zeros((self.hidden_dim, 1))
        hidden_states = [h]

        for x in inputs:
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            hidden_states.append(h)

        y = np.dot(self.Why, h) + self.by
        return y, h, hidden_states

    def backward(self, inputs, targets, hidden_states, loss_grad=1.0):
        """Backpropagation Through Time (BPTT)"""
        # Reset gradients
        self.dWxh.fill(0)
        self.dWhh.fill(0)
        self.dWhy.fill(0)
        self.dbh.fill(0)
        self.dby.fill(0)

        dh_next = np.zeros((self.hidden_dim, 1))

        # Backward pass through time steps
        for t in reversed(range(len(inputs))):
            dy = loss_grad if t == len(inputs) - 1 else np.zeros((self.output_dim, 1))

            # Output layer gradient
            self.dWhy += np.dot(dy, hidden_states[t + 1].T)
            self.dby += dy

            dh = np.dot(self.Why.T, dy) + dh_next

            # Hidden state gradient (backprop through tanh)
            dh_tanh = (1 - hidden_states[t + 1] ** 2) * dh

            # Weight gradients
            self.dWxh += np.dot(dh_tanh, inputs[t].T)
            self.dWhh += np.dot(dh_tanh, hidden_states[t].T)
            self.dbh += dh_tanh

            dh_next = np.dot(self.Whh.T, dh_tanh)

    def update_parameters(self):
        """Gradient descent update"""
        # Clip gradients to prevent explosion
        for dparam in [self.dWxh, self.dWhh, self.dWhy, self.dbh, self.dby]:
            np.clip(dparam, -5, 5, out=dparam)

        self.Wxh -= self.learning_rate * self.dWxh
        self.Whh -= self.learning_rate * self.dWhh
        self.Why -= self.learning_rate * self.dWhy
        self.bh -= self.learning_rate * self.dbh
        self.by -= self.learning_rate * self.dby

class ComparisonModel(nn.Module):
    def __init__(self, cell_type='RNN', input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()
        if cell_type == 'RNN':
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        # Taking the last time step output (Many-to-One)
        return self.fc(out[:, -1, :])


class TimeSeriesRNN(nn.Module):
    """RNN/LSTM/GRU model for time-series prediction"""
    def __init__(self, cell_type='LSTM', input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()
        if cell_type == 'RNN':
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.cell_type = cell_type

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
