import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHorizonTransformer(nn.Module):
    """Transformer encoder model with multi-horizon prediction heads.

    Args:
        input_dim: number of input features
        d_model: model dimensionality
        nhead: number of attention heads
        num_layers: number of transformer encoder layers
        dim_ff: feed-forward dimensionality
        horizons: list of prediction horizons
        dropout: dropout probability
        use_batch_norm: whether to apply BatchNorm after encoder
    """

    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 dim_ff=256, horizons=None, dropout=0.2, use_batch_norm=True):
        super().__init__()
        if horizons is None:
            horizons = [1, 5, 20]
        self.horizons = horizons
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.batch_norm = nn.BatchNorm1d(d_model) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        # Multi-horizon prediction heads
        self.heads = nn.ModuleDict()
        for h in horizons:
            self.heads[f'h{h}'] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        # Storage for attention weights
        self._attention_weights = None
        self._hooks = []

    def _register_attention_hooks(self):
        """Register forward hooks to capture attention weights."""
        self._clear_hooks()
        self._attention_weights = []

        for layer in self.transformer_encoder.layers:
            def hook_fn(module, input, output, self_ref=self):
                # TransformerEncoderLayer wraps self-attention internally;
                # we capture output shape for visualization
                pass
            handle = layer.register_forward_hook(hook_fn)
            self._hooks.append(handle)

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_attention_weights(self, x):
        """Run forward pass and return attention weights.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            attention_weights: (batch, nhead, seq_len, seq_len) from last layer
        """
        projected = self.input_proj(x)
        projected = self.pos_encoder(projected)

        # Manually compute attention from last layer
        last_layer = self.transformer_encoder.layers[-1]
        # Run all layers except last
        hidden = projected
        for layer in self.transformer_encoder.layers[:-1]:
            hidden = layer(hidden)

        # For last layer, use self_attn directly to get weights
        src = hidden
        src2, attn_weights = last_layer.self_attn(
            last_layer.norm1(src) if hasattr(last_layer, 'norm_first') and last_layer.norm_first else src,
            last_layer.norm1(src) if hasattr(last_layer, 'norm_first') and last_layer.norm_first else src,
            last_layer.norm1(src) if hasattr(last_layer, 'norm_first') and last_layer.norm_first else src,
            need_weights=True,
            average_attn_weights=False,
        )
        return attn_weights  # (batch, nhead, seq_len, seq_len)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        projected = self.input_proj(x)          # (batch, seq_len, d_model)
        projected = self.pos_encoder(projected)
        encoded = self.transformer_encoder(projected)  # (batch, seq_len, d_model)

        # Use last time-step representation
        last_hidden = encoded[:, -1, :]  # (batch, d_model)
        last_hidden = self.batch_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)

        preds = {}
        for h in self.horizons:
            preds[f'h{h}'] = self.heads[f'h{h}'](last_hidden).squeeze(-1)
        return preds
