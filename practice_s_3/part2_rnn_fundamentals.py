import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from models import VanillaRNNScratch


def generate_echo_task_data(n_samples=200, seq_len=10, input_dim=5):
    """Task 2.1: Generate echo task - output should echo input after delay"""
    X = []
    y = []

    for _ in range(n_samples):
        # Input: random sequence
        x_seq = np.random.randint(0, 2, (seq_len, input_dim, 1))

        # Output: echo with delay (shift by 2 steps)
        delay = 2
        y_seq = np.zeros((seq_len, 1, 1))
        for t in range(delay, seq_len):
            y_seq[t] = x_seq[t - delay, 0]  # Echo from 2 steps back

        X.append(x_seq)
        y.append(y_seq)

    return X, y


def train_manual_rnn(num_iterations=300, seq_len=8):
    """Task 2.1: Train vanilla RNN on echo task using BPTT"""
    print("="*60)
    print("TASK 2.1: MANUAL RNN TRAINING ON ECHO TASK")
    print("="*60)

    # Generate data
    X_train, y_train = generate_echo_task_data(n_samples=500, seq_len=seq_len, input_dim=5)

    # Initialize model
    model = VanillaRNNScratch(input_dim=5, hidden_dim=20, output_dim=1, learning_rate=0.1)

    losses = []
    print("\nTraining manual RNN...")

    for iteration in range(num_iterations):
        # Random sample
        idx = np.random.randint(0, len(X_train))
        x_seq = [X_train[idx][t] for t in range(seq_len)]
        y_seq = [y_train[idx][t] for t in range(seq_len)]

        # Forward pass
        y_pred, _, hidden_states = model.forward(x_seq)

        # Calculate loss (mean squared error on last output)
        loss = np.mean((y_pred - y_seq[-1]) ** 2)
        losses.append(loss)

        # Backward pass (BPTT)
        loss_grad = 2 * (y_pred - y_seq[-1])
        model.backward(x_seq, y_seq, hidden_states, loss_grad)

        # Update parameters
        model.update_parameters()

        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss:.4f}")

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Manual RNN Training on Echo Task (BPTT)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs('results', exist_ok=True)
    plot_filename = 'results/manual_rnn_training_loss.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    plt.show()

    # Visualize hidden state evolution
    idx = 0
    x_seq = [X_train[idx][t] for t in range(seq_len)]
    _, _, hidden_states = model.forward(x_seq)

    # Convert to array
    hidden_array = np.array([h.squeeze() for h in hidden_states[1:]])

    plt.figure(figsize=(10, 6))
    plt.imshow(hidden_array.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Hidden Unit Activation')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Hidden Unit', fontsize=12)
    plt.title('Manual RNN: Hidden State Evolution Over Time', fontsize=12, fontweight='bold')
    plt.tight_layout()

    heatmap_filename = 'results/manual_rnn_hidden_states_heatmap.png'
    plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved as: {heatmap_filename}")
    plt.show()

    print(f"\nFinal Loss: {losses[-1]:.4f}")
    print("Task 2.1 (Manual RNN) completed successfully!")


def test_manual_rnn():
    print("--- Manual RNN Forward Pass ---")
    model = VanillaRNNScratch(input_dim=5, hidden_dim=10, output_dim=1)
    dummy_input = [np.random.randn(5, 1) for _ in range(3)]  # 3 time steps
    output, last_hidden, hidden_states = model.forward(dummy_input)
    print(f"Output shape: {output.shape}, Hidden shape: {last_hidden.shape}\n")


def analyze_vanishing_gradients(seq_len=100):
    print("="*60)
    print("TASK 2.2: VANISHING GRADIENTS ANALYSIS")
    print("="*60)

    rnn = nn.RNN(1, 10, batch_first=True)
    x = torch.randn(1, seq_len, 1, requires_grad=True)

    out, _ = rnn(x)
    loss = out[:, -1, :].sum()  # Loss on last time step
    loss.backward()

    # Gradient magnitude with respect to input at each time step
    grad_magnitudes = x.grad.abs().squeeze().numpy()

    plt.figure(figsize=(8, 4))
    plt.plot(grad_magnitudes)
    plt.yscale('log')
    plt.title(f"Gradient Magnitude vs. Time Step (Length {seq_len})")
    plt.xlabel("Time Step")
    plt.ylabel("Gradient Norm (Log Scale)")
    plt.grid(True)

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save the plot
    plot_filename = f'results/vanishing_gradients_analysis_seq{seq_len}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    plt.show()


def analyze_vanishing_gradients_multi_length(seq_lengths=[20, 50, 100]):
    """Task 2.2: Test vanishing gradients across multiple sequence lengths"""
    print("="*60)
    print("TASK 2.2: VANISHING GRADIENTS MULTI-LENGTH ANALYSIS")
    print("="*60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    results = {}

    for seq_len, ax in zip(seq_lengths, axes):
        print(f"\nAnalyzing sequence length: {seq_len}")

        rnn = nn.RNN(1, 10, batch_first=True)
        x = torch.randn(1, seq_len, 1, requires_grad=True)

        out, _ = rnn(x)
        loss = out[:, -1, :].sum()
        loss.backward()

        grad_magnitudes = x.grad.abs().squeeze().numpy()
        results[seq_len] = grad_magnitudes

        # Plot
        ax.plot(grad_magnitudes, linewidth=2)
        ax.set_yscale('log')
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Gradient Magnitude (Log)', fontsize=11)
        ax.set_title(f'Seq Length: {seq_len}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Calculate gradient decay
        decay_rate = grad_magnitudes[0] / (grad_magnitudes[-1] + 1e-8)
        print(f"  Gradient decay ratio (first/last): {decay_rate:.2e}")

    plt.tight_layout()

    os.makedirs('results', exist_ok=True)
    plot_filename = 'results/vanishing_gradients_multi_length.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {plot_filename}")
    plt.show()

    # Print analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print("Key Observation: As sequence length increases, gradients vanish more severely")
    print("Gradient decay ratios (first/last gradient magnitude):")
    for seq_len in seq_lengths:
        decay = results[seq_len][0] / (results[seq_len][-1] + 1e-8)
        print(f"  Length {seq_len}: {decay:.2e}")
    print("\nThis demonstrates the vanishing gradient problem in vanilla RNNs.")


def calculate_spectral_radius(seq_length=100):
    """Task 2.2: Calculate spectral radius of RNN weight matrix"""
    print("="*60)
    print("TASK 2.2: SPECTRAL RADIUS ANALYSIS")
    print("="*60)

    # Create RNN and extract hidden-to-hidden weights
    rnn = nn.RNN(1, 10, batch_first=True)

    # Get weight matrix (hidden to hidden)
    W_hh = rnn.weight_hh_l0.data.numpy()

    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(W_hh)
    spectral_radius = np.max(np.abs(eigenvalues))

    print(f"\nSpectral Radius of W_hh: {spectral_radius:.4f}")
    print(f"Weight matrix shape: {W_hh.shape}")
    print(f"Eigenvalue magnitudes (sorted): {sorted(np.abs(eigenvalues), reverse=True)[:5]}")

    # Analysis
    print("\n" + "="*60)
    print("SPECTRAL RADIUS ANALYSIS")
    print("="*60)

    analysis = f"""
Spectral Radius: {spectral_radius:.4f}

INTERPRETATION:
- If ρ(W_hh) < 1: Gradients diminish exponentially (VANISHING GRADIENTS)
- If ρ(W_hh) > 1: Gradients grow exponentially (EXPLODING GRADIENTS)
- If ρ(W_hh) ≈ 1: Gradients remain stable (IDEAL)

Current Status: {'Stable' if 0.95 <= spectral_radius <= 1.05 else ('Vanishing' if spectral_radius < 1 else 'Exploding')}

This explains why vanilla RNNs struggle with long-term dependencies.
LSTMs and GRUs address this by using gating mechanisms that maintain
more stable gradients regardless of sequence length.

For gradient dynamics over {seq_length} time steps:
- Gradient magnitude is multiplied by ρ(W_hh)^{seq_length}
- With ρ = {spectral_radius:.4f}, gradient is scaled by {spectral_radius**seq_length:.2e}
"""

    print(analysis)

    # Plot eigenvalues
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Eigenvalue magnitude plot
    eigenvalue_mags = np.abs(eigenvalues)
    ax1.scatter(range(len(eigenvalue_mags)), sorted(eigenvalue_mags, reverse=True), s=50)
    ax1.axhline(y=1.0, color='r', linestyle='--', label='Unit circle (ρ=1)')
    ax1.set_xlabel('Eigenvalue Index', fontsize=11)
    ax1.set_ylabel('|Eigenvalue|', fontsize=11)
    ax1.set_title('RNN Weight Matrix Eigenvalue Magnitudes', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Eigenvalue complex plane
    ax2.scatter(eigenvalues.real, eigenvalues.imag, s=50, alpha=0.7)
    circle = plt.Circle((0, 0), 1, fill=False, color='r', linestyle='--', label='Unit circle')
    ax2.add_patch(circle)
    ax2.set_xlabel('Real Part', fontsize=11)
    ax2.set_ylabel('Imaginary Part', fontsize=11)
    ax2.set_title('Eigenvalues in Complex Plane', fontsize=12, fontweight='bold')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    os.makedirs('results', exist_ok=True)
    plot_filename = 'results/spectral_radius_analysis.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    plt.show()


if __name__ == "__main__":
    test_manual_rnn()
    analyze_vanishing_gradients()
    analyze_vanishing_gradients_multi_length()
    calculate_spectral_radius()
