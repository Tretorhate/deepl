import torch
import torch.nn as nn


class MultiHorizonLSTM(nn.Module):
    """LSTM/GRU model with multi-horizon prediction heads.

    Args:
        input_dim: number of input features
        hidden_dim: hidden state dimensionality
        num_layers: number of recurrent layers
        horizons: list of prediction horizons (e.g. [1, 5, 20])
        dropout: dropout probability between recurrent layers
        use_batch_norm: whether to apply BatchNorm after recurrent output
        cell_type: 'LSTM' or 'GRU'
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, horizons=None,
                 dropout=0.2, use_batch_norm=True, cell_type='LSTM'):
        super().__init__()
        if horizons is None:
            horizons = [1, 5, 20]
        self.horizons = horizons
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type
        self.use_batch_norm = use_batch_norm

        rnn_cls = nn.LSTM if cell_type == 'LSTM' else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.batch_norm = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        # Separate prediction head per horizon
        self.heads = nn.ModuleDict()
        for h in horizons:
            self.heads[f'h{h}'] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        rnn_out, _ = self.rnn(x)          # (batch, seq_len, hidden_dim)
        last_hidden = rnn_out[:, -1, :]   # (batch, hidden_dim)
        last_hidden = self.batch_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)

        preds = {}
        for h in self.horizons:
            preds[f'h{h}'] = self.heads[f'h{h}'](last_hidden).squeeze(-1)
        return preds
