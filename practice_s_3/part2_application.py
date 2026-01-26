import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from models import TimeSeriesRNN


def generate_time_series_data(n_points=300, noise_level=0.1):
    """Generate synthetic time-series data (sine wave with noise)"""
    t = np.linspace(0, 4 * np.pi, n_points)
    # Sine wave
    y = np.sin(t) + np.random.normal(0, noise_level, n_points)
    return t, y


def create_sequences(data, seq_length=20):
    """Create sequences from time series data (sliding window)"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def baseline_moving_average(y_test, window_size=5):
    """Simple moving average baseline"""
    predictions = []
    for i in range(len(y_test)):
        if i < window_size:
            pred = np.mean(y_test[:i + 1])
        else:
            pred = np.mean(y_test[i - window_size:i])
        predictions.append(pred)
    return np.array(predictions)


def train_rnn_model(X_train, y_train, X_val, y_val, cell_type='LSTM', epochs=25, batch_size=32):
    """Train RNN/LSTM/GRU model"""
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(-1)
    X_val_t = torch.FloatTensor(X_val).unsqueeze(-1)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(-1)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = TimeSeriesRNN(cell_type=cell_type, input_dim=1, hidden_dim=32, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_t)
            val_loss = criterion(y_val_pred, y_val_t).item()
            val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}")

    return model, train_losses, val_losses


def run_time_series_prediction():
    """Task 2.4: Time-Series Prediction Application"""
    print("="*60)
    print("TASK 2.4: TIME-SERIES PREDICTION APPLICATION")
    print("="*60)

    # Generate data
    print("\nGenerating synthetic time-series data...")
    t, y = generate_time_series_data(n_points=500, noise_level=0.1)

    # Split into train/val/test
    train_size = int(0.6 * len(y))
    val_size = int(0.2 * len(y))

    y_train = y[:train_size]
    y_val = y[train_size:train_size + val_size]
    y_test = y[train_size + val_size:]

    # Create sequences
    seq_length = 15
    X_train, Y_train = create_sequences(y_train, seq_length)
    X_val, Y_val = create_sequences(y_val, seq_length)
    X_test, Y_test = create_sequences(y_test, seq_length)

    print(f"Training sequences: {X_train.shape[0]}")
    print(f"Validation sequences: {X_val.shape[0]}")
    print(f"Test sequences: {X_test.shape[0]}")

    # Train models
    models_to_train = ['RNN', 'LSTM', 'GRU']
    trained_models = {}
    predictions = {}
    metrics = {}

    print("\n" + "="*60)
    print("Training RNN Models")
    print("="*60)

    for cell_type in models_to_train:
        print(f"\nTraining {cell_type}...")
        model, train_losses, val_losses = train_rnn_model(
            X_train, Y_train, X_val, Y_val, cell_type=cell_type, epochs=50
        )

        # Evaluate on test set
        model.eval()
        X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)
        with torch.no_grad():
            y_pred = model(X_test_t).numpy().squeeze()

        trained_models[cell_type] = model
        predictions[cell_type] = y_pred

        # Calculate metrics
        mse = np.mean((y_pred - Y_test) ** 2)
        mae = np.mean(np.abs(y_pred - Y_test))

        metrics[cell_type] = {'mse': mse, 'mae': mae}
        print(f"  Test MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Baseline (Moving Average)
    print(f"\nTraining Moving Average baseline...")
    y_combined = np.concatenate([y_train, y_val])
    y_ma = baseline_moving_average(y_combined, window_size=5)
    y_ma_test = y_ma[-len(Y_test):]
    ma_mse = np.mean((y_ma_test - Y_test) ** 2)
    ma_mae = np.mean(np.abs(y_ma_test - Y_test))
    metrics['Moving Average'] = {'mse': ma_mse, 'mae': ma_mae}
    predictions['Moving Average'] = y_ma_test
    print(f"  Test MSE: {ma_mse:.4f}, MAE: {ma_mae:.4f}")

    # Plot 1: Time-Series Predictions
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    plt.figure(figsize=(14, 6))

    # Plot actual test data
    test_start_idx = len(y[:train_size + val_size + seq_length])
    t_test = t[test_start_idx:test_start_idx + len(Y_test)]

    plt.plot(t_test, Y_test, 'ko-', label='Actual', linewidth=2, markersize=4)

    # Plot predictions
    colors = {'RNN': 'blue', 'LSTM': 'green', 'GRU': 'orange', 'Moving Average': 'red'}
    for cell_type in models_to_train + ['Moving Average']:
        plt.plot(t_test, predictions[cell_type], '--', label=cell_type, linewidth=1.5, alpha=0.7)

    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Time-Series Prediction: Actual vs Predicted', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs('results', exist_ok=True)
    plot_filename = 'results/time_series_predictions.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    plt.show()

    # Plot 2: Error Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    models_names = models_to_train + ['Moving Average']
    mse_values = [metrics[m]['mse'] for m in models_names]
    mae_values = [metrics[m]['mae'] for m in models_names]

    ax1.bar(models_names, mse_values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    ax1.set_ylabel('MSE', fontsize=11)
    ax1.set_title('Mean Squared Error Comparison', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(models_names, mae_values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    ax2.set_ylabel('MAE', fontsize=11)
    ax2.set_title('Mean Absolute Error Comparison', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    plot_filename2 = 'results/time_series_error_comparison.png'
    plt.savefig(plot_filename2, dpi=300, bbox_inches='tight')
    print(f"Error comparison plot saved as: {plot_filename2}")
    plt.show()

    # Print comparison table
    print("\n" + "="*60)
    print("TIME-SERIES PREDICTION COMPARISON TABLE")
    print("="*60)
    print(f"{'Model':<20} {'MSE':<15} {'MAE':<15}")
    print("-"*50)

    for model_name in models_names:
        mse = metrics[model_name]['mse']
        mae = metrics[model_name]['mae']
        print(f"{model_name:<20} {mse:<15.4f} {mae:<15.4f}")

    # Find best model
    best_model = min(metrics.items(), key=lambda x: x[1]['mse'])[0]
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print(f"Best performing model: {best_model} (lowest MSE)")

    analysis = """
KEY FINDINGS:

1. Model Performance:
   - RNN/LSTM/GRU: Deep learning models capture temporal patterns
   - Moving Average: Simple baseline for comparison

2. Advantages of RNN models:
   - Can learn complex temporal patterns
   - Adaptable to various sequence lengths
   - Better for non-stationary data

3. Advantages of Moving Average:
   - Simple and interpretable
   - Fast to compute
   - Good baseline for stationary data

4. Practical Recommendations:
   - Use LSTM/GRU for complex time-series with long-term dependencies
   - Use RNN for simpler patterns
   - Compare against baseline (moving average) always
   - Consider ensemble approaches for production systems

5. Limitations of this experiment:
   - Simple synthetic data (sine wave)
   - Short prediction horizon (one step ahead)
   - Small dataset size
   - For production: use real data, cross-validation, hyperparameter tuning
"""

    print(analysis)

    print("\nTask 2.4 completed successfully!")


if __name__ == "__main__":
    run_time_series_prediction()
