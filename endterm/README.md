# Multi-Horizon Financial Predictor

Deep Learning end-term project (Track 2: Predictive Intelligence for Time-Series, 100 pts).
Integrates three core topics: Sequences (LSTM/GRU), Transformers, and Regularization.

## Project Structure

```
endterm/
├── config.py                  # Hyperparameters and execution modes
├── main.py                    # Entry point with interactive menu
├── requirements.txt           # Additional dependencies
├── data/
│   └── data_loader.py         # Download, feature engineering, windowing, splits
├── models/
│   ├── lstm_model.py          # LSTM and GRU models
│   └── transformer_model.py   # Transformer encoder model
├── training/
│   ├── trainer.py             # Training loop, early stopping, LR scheduling
│   └── ensemble.py            # Multi-seed ensemble with uncertainty estimation
├── evaluation/
│   ├── metrics.py             # MSE, RMSE, MAE, MAPE, directional accuracy, R-squared
│   └── ablation.py            # Ablation study runner
├── visualization/
│   └── plots.py               # All visualization functions
└── results/                   # Auto-created output directory
```

## Setup

Requires Python 3.8 or later. Install dependencies inside your virtual environment:

```
pip install -r requirements.txt
```

Core dependencies: PyTorch, yfinance, ta, pandas, numpy, scikit-learn, matplotlib, seaborn.

## Usage

### Interactive Mode

```
python main.py
```

On first run, you'll be prompted to select an execution mode (QUICK, HYBRID, or FULL). Then presents a menu:

```
1. Download & Prepare Data
2. Train LSTM Model
3. Train GRU Model
4. Train Transformer Model
5. Run Ensemble Prediction (with uncertainty)
6. Run Ablation Studies
7. Generate All Visualizations
8. Run Full Pipeline (1-7)
9. Change Execution Mode
0. Exit
```

Option 9 lets you switch modes anytime without restarting.

### Command-Line Mode

Run a specific menu option directly without the interactive prompt:

```
python main.py --mode QUICK --run 8
```

Arguments:

- `--mode`: Execution mode. One of `QUICK`, `HYBRID`, or `FULL`. Defaults to `QUICK`.
- `--run`: Menu option number to execute (1-8).

## Execution Modes

| Mode   | Epochs | Tickers                          | Ensemble Seeds | Ablation Epochs | Patience |
|--------|--------|----------------------------------|----------------|-----------------|----------|
| QUICK  | 30     | AAPL                             | 2              | 15              | 20       |
| HYBRID | 100    | AAPL, MSFT, GOOGL                | 3              | 50              | 30       |
| FULL   | 200    | AAPL, MSFT, GOOGL, AMZN, ^GSPC  | 5              | 100             | 40       |

QUICK mode is intended for fast verification. FULL mode runs all experiments end-to-end. Patience is the number of epochs without improvement before early stopping activates (after a warmup period).

## Models

### LSTM / GRU

- 2-layer recurrent network (LSTM or GRU, selectable via `cell_type`)
- Hidden dimension: 128
- Optional batch normalization after the recurrent output
- Dropout between layers (default 0.2)
- Separate linear prediction heads for each forecast horizon

### Transformer

- Input projection to d_model=64
- Sinusoidal positional encoding
- 2-layer TransformerEncoder with 4 attention heads
- Feed-forward dimension: 256
- Optional batch normalization
- Attention weight extraction for visualization
- Same multi-horizon output heads as LSTM/GRU

All three models output predictions as a dictionary with keys `h1`, `h5`, `h20` corresponding to 1-day, 5-day, and 20-day return forecasts.

## Data Pipeline

- Downloads OHLCV data via yfinance (2018-2024)
- Engineers 25+ features: returns, log returns, SMA (5/10/20/50), EMA (12/26), RSI-14, MACD, Bollinger Bands, volume change, ATR-14, cyclical day-of-week and month encodings
- Chronological train/val/test split (70/15/15) with no data leakage (scaler fit on train only)
- Sliding window sequences: input shape (N, 60, features), target shape (N, 3)

## Training

- Loss: MSE summed across all horizons
- Optimizer: Adam with weight decay (1e-4)
- LR scheduler: ReduceLROnPlateau (factor=0.5, patience=10)
- Warmup period: Early stopping patience counter does not start until after 20% of total epochs (or minimum 10 epochs), allowing the model to stabilize before evaluation
- Early stopping with mode-dependent patience (20/30/40 for QUICK/HYBRID/FULL)
- Gradient clipping (max norm 1.0)
- Best model checkpoint restored after training

## Ensemble and Uncertainty

- Trains N models (mode-dependent) with different random seeds
- Aggregates predictions: mean and standard deviation
- 95% confidence intervals: mean +/- 1.96 * std

## Ablation Studies

Seven experiments, each varying one factor:

1. Model type: LSTM vs GRU vs Transformer
2. Dropout rate: 0.0, 0.1, 0.2, 0.3, 0.5
3. Batch normalization: on vs off
4. Transformer attention depth: 1 layer vs 2 layers
5. Weight decay: 0, 1e-5, 1e-4, 1e-3
6. Single-horizon (h1 only) vs multi-horizon
7. Window sizes: 30, 60, 90 (requires data re-creation, handled separately)

Results are saved to `results/ablation_results.csv`.

## Metrics

All evaluation uses the following metrics per horizon:

- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy (percentage of correct direction predictions)
- R-squared

## Visualizations

All plots are saved to timestamped result directories under `results/`:

```
results/
├── results_QUICK_20250209_131150/
│   ├── training_curves.png
│   ├── LSTM_predictions_h1.png
│   ├── LSTM_residuals_h1.png
│   ├── ablation_results.csv
│   ├── ablation_results.png
│   ├── model_comparison.png
│   └── ... (all other outputs)
├── results_QUICK_20250209_131200/
│   └── ... (next run)
```

Each run creates a new timestamped subdirectory named `results_{MODE}_{YYYYMMDD}_{HHMMSS}` to keep experiments organized.

Generated plots include:

- Training and validation loss curves for all models
- Prediction vs actual plots with 95% confidence bands
- Transformer attention heatmaps (per head)
- Ablation study bar charts
- Gradient-based feature importance (top 20 features)
- Model comparison across horizons (RMSE, MAE, directional accuracy, R-squared)
- Residual analysis (residuals over time, distribution, predicted vs actual scatter)

## Configuration

All hyperparameters are centralized in `config.py`. Key settings:

- `RANDOM_SEED`: 42 (for reproducibility)
- `DEVICE`: auto-detected (CUDA if available, otherwise CPU)
- `CURRENT_MODE`: set via CLI or modified directly in config
- Model, training, ensemble, and ablation configs are separate dictionaries

## Reproducibility

Setting the same seed produces the same results. The random seed is applied to PyTorch, NumPy, and CUDA (if available) before each training run.
