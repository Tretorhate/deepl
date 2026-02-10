import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


def download_stock_data(tickers, start, end):
    """Download OHLCV data for given tickers via yfinance."""
    all_data = {}
    for ticker in tickers:
        print(f"  Downloading {ticker}...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if hasattr(df.columns, 'levels') and df.columns.nlevels > 1:
            df.columns = df.columns.droplevel(1)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.dropna(inplace=True)
        all_data[ticker] = df
        print(f"    {ticker}: {len(df)} trading days")
    return all_data


def engineer_features(df):
    """Add technical indicators and engineered features to OHLCV dataframe."""
    df = df.copy()

    # Returns
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Simple Moving Averages
    for w in [5, 10, 20, 50]:
        df[f'SMA_{w}'] = df['Close'].rolling(window=w).mean()

    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # RSI (14)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = sma20 + 2 * std20
    df['BB_Lower'] = sma20 - 2 * std20
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (sma20 + 1e-10)

    # Volume change
    df['Volume_Change'] = df['Volume'].pct_change()

    # ATR (14)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(14).mean()

    # Cyclical time features
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 5)
    df['Month_Sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # Drop NaN rows created by rolling features
    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    return df


def normalize_data(train_df, val_df, test_df):
    """Fit StandardScaler on train, transform all splits. Returns arrays + scaler."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)
    test_scaled = scaler.transform(test_df.values)
    return train_scaled, val_scaled, test_scaled, scaler


def create_sequences(data, seq_len, horizons, close_col_idx=3):
    """Create sliding window sequences.

    Args:
        data: scaled numpy array (N, features)
        seq_len: input window length
        horizons: list of prediction horizons [1, 5, 20]
        close_col_idx: index of Close price in features

    Returns:
        X: (num_samples, seq_len, features)
        Y: (num_samples, len(horizons)) — future returns for each horizon
    """
    max_horizon = max(horizons)
    X, Y = [], []

    for i in range(seq_len, len(data) - max_horizon):
        X.append(data[i - seq_len:i])
        targets = []
        current_close = data[i - 1, close_col_idx]
        for h in horizons:
            future_close = data[i - 1 + h, close_col_idx]
            ret = (future_close - current_close) / (abs(current_close) + 1e-10)
            targets.append(ret)
        Y.append(targets)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def get_dataloaders(ticker_df, config):
    """Build train/val/test DataLoaders from a single ticker DataFrame.

    Args:
        ticker_df: raw OHLCV DataFrame (already downloaded)
        config: DATA_CONFIG dict

    Returns:
        train_loader, val_loader, test_loader, scaler, feature_names
    """
    df = engineer_features(ticker_df)
    feature_names = list(df.columns)

    n = len(df)
    train_end = int(n * config['train_ratio'])
    val_end = int(n * (config['train_ratio'] + config['val_ratio']))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    train_scaled, val_scaled, test_scaled, scaler = normalize_data(train_df, val_df, test_df)

    close_col_idx = feature_names.index('Close')

    seq_len = config['seq_len']
    horizons = config['horizons']

    X_train, Y_train = create_sequences(train_scaled, seq_len, horizons, close_col_idx)
    X_val, Y_val = create_sequences(val_scaled, seq_len, horizons, close_col_idx)
    X_test, Y_test = create_sequences(test_scaled, seq_len, horizons, close_col_idx)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test))

    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"    Sequences -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"    Features: {len(feature_names)}, Seq len: {seq_len}")

    return train_loader, val_loader, test_loader, scaler, feature_names
