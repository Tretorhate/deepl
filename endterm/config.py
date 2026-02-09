import torch


RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Execution modes ──────────────────────────────────────────────────────────
MODES = {
    'QUICK': {
        'epochs': 30,
        'tickers': ['AAPL'],
        'num_seeds': 2,
        'ablation_epochs': 15,
        'patience': 20,
    },
    'HYBRID': {
        'epochs': 100,
        'tickers': ['AAPL', 'MSFT', 'GOOGL'],
        'num_seeds': 3,
        'ablation_epochs': 50,
        'patience': 30,
    },
    'FULL': {
        'epochs': 200,
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', '^GSPC'],
        'num_seeds': 5,
        'ablation_epochs': 100,
        'patience': 40,
    },
}

CURRENT_MODE = 'QUICK'

# ── Data config ──────────────────────────────────────────────────────────────
DATA_CONFIG = {
    'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', '^GSPC'],
    'start_date': '2018-01-01',
    'end_date': '2024-12-31',
    'seq_len': 60,
    'horizons': [1, 5, 20],
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
}

# ── LSTM config ──────────────────────────────────────────────────────────────
LSTM_CONFIG = {
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'bidirectional': False,
    'use_batch_norm': True,
}

# ── GRU config ───────────────────────────────────────────────────────────────
GRU_CONFIG = {
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'bidirectional': False,
    'use_batch_norm': True,
}

# ── Transformer config ──────────────────────────────────────────────────────
TRANSFORMER_CONFIG = {
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256,
    'dropout': 0.2,
    'use_batch_norm': True,
}

# ── Training config ─────────────────────────────────────────────────────────
TRAINING_CONFIG = {
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'patience': 15,
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_factor': 0.5,
    'scheduler_patience': 10,
}

# ── Ensemble config ─────────────────────────────────────────────────────────
ENSEMBLE_CONFIG = {
    'num_seeds': 5,
    'base_seed': 42,
}

# ── Ablation config ─────────────────────────────────────────────────────────
ABLATION_CONFIG = {
    'dropout_rates': [0.0, 0.1, 0.2, 0.3, 0.5],
    'window_sizes': [30, 60, 90],
    'weight_decays': [0, 1e-5, 1e-4, 1e-3],
    'model_types': ['LSTM', 'GRU', 'Transformer'],
    'batch_norm_toggle': [True, False],
    'attention_toggle': [True, False],
}


def get_mode_config():
    """Return effective config merged with current mode settings."""
    mode = MODES[CURRENT_MODE]
    return {
        'epochs': mode['epochs'],
        'tickers': mode['tickers'],
        'num_seeds': mode['num_seeds'],
        'ablation_epochs': mode['ablation_epochs'],
        'patience': mode['patience'],
    }
