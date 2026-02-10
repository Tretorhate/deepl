"""
Text Classification Models
Implements LSTM/GRU and Transformer-based text classifiers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMTextClassifier(nn.Module):
    """
    LSTM/GRU-based text classifier for medical documents.

    Architecture:
    - Embedding layer
    - Optional dropout
    - LSTM/GRU layer(s)
    - Dense classification head
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        num_layers=2,
        num_classes=4,
        dropout=0.3,
        cell_type='LSTM',
        bidirectional=True,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden states
            num_layers: Number of RNN layers
            num_classes: Number of output classes
            dropout: Dropout rate
            cell_type: 'LSTM' or 'GRU'
            bidirectional: Whether to use bidirectional RNN
        """
        super(LSTMTextClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # RNN layer
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:  # GRU
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )

        # Calculate output size from RNN
        rnn_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(rnn_output_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: Token indices (batch_size, seq_len)
            attention_mask: Mask for padded tokens (batch_size, seq_len)

        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)

        # RNN forward pass
        if self.cell_type == 'LSTM':
            rnn_output, (hidden, cell) = self.rnn(embedded)
        else:  # GRU
            rnn_output, hidden = self.rnn(embedded)

        # Use last hidden state (take last layer, both directions if bidirectional)
        if self.bidirectional:
            hidden_last = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden_last = hidden[-1]  # (batch_size, hidden_dim)

        # Classification head
        hidden_last = self.dropout(hidden_last)
        logits = self.fc1(hidden_last)  # (batch_size, hidden_dim // 2)
        logits = F.relu(logits)
        logits = self.dropout(logits)
        logits = self.fc2(logits)  # (batch_size, num_classes)

        return logits


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.

    Based on "Attention is All You Need" paper.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """Add positional encoding to embeddings."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerTextClassifier(nn.Module):
    """
    Transformer-based text classifier for medical documents.

    Architecture:
    - Embedding layer
    - Positional encoding
    - Transformer encoder layers
    - Global average pooling
    - Dense classification head
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        num_classes=4,
        dropout=0.1,
        max_seq_len=128,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            d_model: Dimension of model (must be divisible by nhead)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feed-forward network
            num_classes: Number of output classes
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super(TransformerTextClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Project embedding to d_model dimension if needed
        if embedding_dim != d_model:
            self.embedding_projection = nn.Linear(embedding_dim, d_model)
        else:
            self.embedding_projection = nn.Identity()

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu',
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, dim_feedforward // 2)
        self.fc2 = nn.Linear(dim_feedforward // 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: Token indices (batch_size, seq_len)
            attention_mask: Mask for padded tokens (batch_size, seq_len)

        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        # Create attention mask for transformer
        # Transformer expects mask where True means "ignore this position"
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding_projection(embedded)  # (batch_size, seq_len, d_model)

        # Positional encoding
        encoded = self.positional_encoding(embedded)

        # Transformer encoder
        transformer_output = self.transformer_encoder(
            encoded, src_key_padding_mask=src_key_padding_mask
        )  # (batch_size, seq_len, d_model)

        # Global average pooling (pool over sequence dimension)
        # Only pool over non-padded positions if mask is available
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(transformer_output.size()).float()
            sum_output = (transformer_output * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1)
            pooled = sum_output / count
        else:
            pooled = transformer_output.mean(dim=1)  # (batch_size, d_model)

        # Classification head
        hidden = self.dropout(pooled)
        hidden = self.fc1(hidden)  # (batch_size, dim_feedforward // 2)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)  # (batch_size, num_classes)

        return logits


if __name__ == '__main__':
    # Test LSTM
    print("Testing LSTM classifier...")
    lstm_model = LSTMTextClassifier(
        vocab_size=5000,
        embedding_dim=100,
        hidden_dim=128,
        num_layers=2,
        num_classes=4,
        dropout=0.3,
        cell_type='LSTM',
        bidirectional=True,
    )

    # Test forward pass
    batch_size = 16
    seq_len = 128
    input_ids = torch.randint(0, 5000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    output = lstm_model(input_ids, attention_mask)
    print(f"LSTM output shape: {output.shape}")
    assert output.shape == (batch_size, 4), f"Expected shape {(batch_size, 4)}, got {output.shape}"
    print("LSTM test passed!\n")

    # Test Transformer
    print("Testing Transformer classifier...")
    transformer_model = TransformerTextClassifier(
        vocab_size=5000,
        embedding_dim=100,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        num_classes=4,
        dropout=0.1,
    )

    output = transformer_model(input_ids, attention_mask)
    print(f"Transformer output shape: {output.shape}")
    assert output.shape == (batch_size, 4), f"Expected shape {(batch_size, 4)}, got {output.shape}"
    print("Transformer test passed!\n")

    # Test GRU variant
    print("Testing GRU classifier...")
    gru_model = LSTMTextClassifier(
        vocab_size=5000,
        embedding_dim=100,
        hidden_dim=128,
        num_layers=2,
        num_classes=4,
        dropout=0.3,
        cell_type='GRU',
        bidirectional=True,
    )

    output = gru_model(input_ids, attention_mask)
    print(f"GRU output shape: {output.shape}")
    assert output.shape == (batch_size, 4), f"Expected shape {(batch_size, 4)}, got {output.shape}"
    print("GRU test passed!\n")

    print("All model tests passed!")
