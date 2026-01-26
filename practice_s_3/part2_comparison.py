import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from models import ComparisonModel


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_model_comparison():
    """Basic model comparison"""
    print("="*60)
    print("TASK 2.3: RNN vs LSTM vs GRU COMPARISON (BASIC)")
    print("="*60)

    seq_len = 15
    X = torch.randn(80, seq_len, 1)
    y = X.sum(dim=1)

    results = {}
    param_counts = {}

    for cell in ['RNN', 'LSTM', 'GRU']:
        print(f"\nTraining {cell}...")
        model = ComparisonModel(cell_type=cell, hidden_dim=32)
        param_counts[cell] = count_parameters(model)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        losses = []
        for epoch in range(30):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        results[cell] = losses
        print(f"  Parameters: {param_counts[cell]}")

    # Plot convergence curves
    plt.figure(figsize=(12, 5))

    for cell, loss_vals in results.items():
        plt.plot(loss_vals, label=f'{cell} (params: {param_counts[cell]})', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Convergence Comparison: RNN vs LSTM vs GRU', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save the plot
    plot_filename = 'results/rnn_convergence_comparison.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {plot_filename}")
    plt.show()

    print("\nTask 2.3 (Basic) completed successfully!")


def run_ablation_study(seq_lengths=[15, 30, 50], epochs=20):
    """Task 2.3: Ablation study on sequence lengths"""
    print("="*60)
    print("TASK 2.3: ABLATION STUDY (SEQUENCE LENGTH)")
    print("="*60)

    models_to_test = ['RNN', 'LSTM', 'GRU']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    results_summary = {}

    for seq_idx, seq_len in enumerate(seq_lengths):
        print(f"\n--- Sequence Length: {seq_len} ---")

        # Generate data
        X = torch.randn(60, seq_len, 1)
        y = X.sum(dim=1)

        for cell in models_to_test:
            model = ComparisonModel(cell_type=cell, hidden_dim=32)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            losses = []
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            key = f"{cell}_seq{seq_len}"
            results_summary[key] = losses

            # Plot on corresponding subplot
            axes[seq_idx].plot(losses, label=cell, linewidth=2, marker='o', markersize=3)
            print(f"  {cell}: Final loss = {losses[-1]:.4f}")

        axes[seq_idx].set_xlabel('Epoch', fontsize=11)
        axes[seq_idx].set_ylabel('Loss (MSE)', fontsize=11)
        axes[seq_idx].set_title(f'Sequence Length: {seq_len}', fontsize=11, fontweight='bold')
        axes[seq_idx].legend()
        axes[seq_idx].grid(True, alpha=0.3)

    plt.suptitle('Ablation Study: Model Comparison Across Sequence Lengths',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs('results', exist_ok=True)
    plot_filename = 'results/ablation_study_sequence_lengths.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nAblation study plot saved as: {plot_filename}")
    plt.show()

    print("\nAblation Study completed successfully!")


def create_comprehensive_comparison():
    """Task 2.3: Comprehensive model comparison with metrics"""
    print("="*60)
    print("TASK 2.3: COMPREHENSIVE MODEL COMPARISON")
    print("="*60)

    models_to_test = ['RNN', 'LSTM', 'GRU']
    seq_len = 30
    epochs = 15

    # Generate data
    X = torch.randn(100, seq_len, 1)
    y = X.sum(dim=1)

    results = {}

    print(f"\nTesting on sequence length: {seq_len}, epochs: {epochs}\n")

    for cell in models_to_test:
        print(f"Training {cell}...")

        model = ComparisonModel(cell_type=cell, hidden_dim=32)
        params = count_parameters(model)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Time the training
        start_time = time.time()

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        training_time = time.time() - start_time
        final_loss = loss.item()

        results[cell] = {
            'parameters': params,
            'training_time': training_time,
            'final_loss': final_loss
        }

        print(f"  Parameters: {params}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Final loss: {final_loss:.4f}\n")

    # Print comparison table
    print("="*60)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*60)
    print(f"{'Model':<15} {'Parameters':<15} {'Time (sec)':<15} {'Final Loss':<15}")
    print("-"*60)

    for cell in models_to_test:
        print(f"{cell:<15} {results[cell]['parameters']:<15} {results[cell]['training_time']:<15.2f} {results[cell]['final_loss']:<15.4f}")

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    analysis = """
Model Selection Guide:

1. RNN (Vanilla Recurrent Neural Network):
   - Simplest architecture
   - Fewest parameters
   - Suffers from vanishing gradient problem
   - Best for: Simple sequences, short-term dependencies

2. LSTM (Long Short-Term Memory):
   - Gating mechanism (forget, input, output gates)
   - More parameters than RNN
   - Better gradient flow (mitigates vanishing gradients)
   - Best for: Long-term dependencies, complex patterns

3. GRU (Gated Recurrent Unit):
   - Simplified LSTM with reset and update gates
   - Fewer parameters than LSTM
   - Similar performance to LSTM with less computation
   - Best for: Balance between performance and efficiency

Observations on this dataset (sequence sum task):
- LSTM typically shows best convergence (handles long-term dependencies)
- GRU offers similar performance with fewer parameters
- RNN may struggle if sequences become longer
- All models can handle this simple sum task reasonably well
"""

    print(analysis)

    print("\nComprehensive Comparison completed successfully!")


if __name__ == "__main__":
    run_model_comparison()
    run_ablation_study()
    create_comprehensive_comparison()
