"""
Training scripts for MLP experiments.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from section1.mlp import MLP, cross_entropy_loss, softmax
from utils.data_loader import load_mnist_numpy
from utils.visualization import plot_training_curves
from config import SECTION1_CONFIG, RESULTS_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)


def train_mlp(X_train, y_train, X_val, y_val, model, epochs, learning_rate, batch_size):
    """Train MLP model."""
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    n_train = X_train.shape[0]
    n_batches = n_train // batch_size
    
    for epoch in range(epochs):
        # Training
        epoch_loss = 0
        correct = 0
        
        # Shuffle data
        indices = np.random.permutation(n_train)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_train)
            
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # Forward pass
            y_pred = model.forward(X_batch)
            loss, grad = cross_entropy_loss(y_pred, y_batch)
            
            # Backward pass
            grads = model.backward(grad)
            model.update(grads, learning_rate)
            
            epoch_loss += loss
            predictions = np.argmax(softmax(y_pred), axis=1)
            correct += np.sum(predictions == y_batch)
        
        train_loss = epoch_loss / n_batches
        train_acc = correct / n_train
        
        # Validation
        y_val_pred = model.forward(X_val)
        val_loss, _ = cross_entropy_loss(y_val_pred, y_val)
        val_predictions = np.argmax(softmax(y_val_pred), axis=1)
        val_acc = np.mean(val_predictions == y_val)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
    
    return train_losses, val_losses, train_accs, val_accs


def run_section1_experiments():
    """Run all Section 1 experiments."""
    from config import get_section_mode, is_quick_mode, is_hybrid_mode
    # Get mode for this section
    section_mode = get_section_mode(1)
    QUICK_MODE = is_quick_mode(1)
    HYBRID_MODE = is_hybrid_mode(1)
    
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist_numpy()
    
    # Adjust dataset size based on mode
    if QUICK_MODE:
        print("QUICK MODE: Using reduced dataset size")
        X_train = X_train[:10000]  # Use only 10k samples instead of 50k
        y_train = y_train[:10000]
        X_test = X_test[:2000]
        y_test = y_test[:2000]
    elif HYBRID_MODE:
        print("HYBRID MODE: Using moderate dataset size (25k samples)")
        X_train = X_train[:25000]  # Use 25k samples for faster training
        y_train = y_train[:25000]
        X_test = X_test[:5000]
        y_test = y_test[:5000]
    
    # Split training into train/val
    if QUICK_MODE:
        val_size = min(2000, len(X_train) // 5)
    elif HYBRID_MODE:
        val_size = 5000
    else:
        val_size = 10000
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    # Experiment 1: Compare activation functions
    print("=" * 50)
    print("Experiment 1: Comparing Activation Functions")
    print("=" * 50)
    
    # Use all activations in hybrid/full mode, fewer in quick mode
    if QUICK_MODE:
        activations = ['relu', 'tanh']
        print("QUICK MODE: Testing only ReLU and Tanh activations")
    else:
        activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu']  # All 4 for requirements
        if HYBRID_MODE:
            print("HYBRID MODE: Testing all 4 activation functions")
    results = {}
    
    for activation in activations:
        print(f"\nTraining with {activation} activation...")
        model = MLP(
            input_size=784,
            hidden_sizes=SECTION1_CONFIG['hidden_sizes'],
            output_size=10,
            activation=activation
        )
        
        train_losses, val_losses, train_accs, val_accs = train_mlp(
            X_train, y_train, X_val, y_val,
            model, SECTION1_CONFIG['epochs'],
            SECTION1_CONFIG['learning_rate'],
            SECTION1_CONFIG['batch_size']
        )
        
        results[activation] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
        
        # Test accuracy
        y_test_pred = model.forward(X_test)
        test_predictions = np.argmax(softmax(y_test_pred), axis=1)
        test_acc = np.mean(test_predictions == y_test)
        print(f"Test Accuracy: {test_acc:.4f}")
    
    # Plot comparison
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for activation in activations:
        epochs = range(1, len(results[activation]['val_losses']) + 1)
        ax1.plot(epochs, results[activation]['val_losses'], label=activation)
        ax2.plot(epochs, results[activation]['val_accs'], label=activation)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Activation Function Comparison - Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Activation Function Comparison - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'section1', 'activation_comparison.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved activation comparison to {save_path}")
    plt.close()
    
    # Experiment 2: XOR problem
    print("\n" + "=" * 50)
    print("Experiment 2: XOR Problem")
    print("=" * 50)
    
    # XOR dataset
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    model_xor = MLP(
        input_size=2,
        hidden_sizes=[4, 4],
        output_size=2,
        activation='tanh'
    )
    
    print("Training on XOR problem...")
    if QUICK_MODE:
        xor_epochs = 200
    elif HYBRID_MODE:
        xor_epochs = 300  # Reduced for faster execution (minimal quality mode)
    else:
        xor_epochs = 1000
    for epoch in range(xor_epochs):
        y_pred = model_xor.forward(X_xor)
        loss, grad = cross_entropy_loss(y_pred, y_xor)
        grads = model_xor.backward(grad)
        model_xor.update(grads, learning_rate=0.1)
        
        if (epoch + 1) % (50 if QUICK_MODE else 200) == 0:
            predictions = np.argmax(softmax(y_pred), axis=1)
            acc = np.mean(predictions == y_xor)
            print(f"Epoch {epoch+1}/{xor_epochs} - Loss: {loss:.4f}, Acc: {acc:.4f}")
    
    # Test XOR
    y_xor_pred = model_xor.forward(X_xor)
    xor_predictions = np.argmax(softmax(y_xor_pred), axis=1)
    print(f"\nXOR Predictions: {xor_predictions}")
    print(f"XOR True Labels: {y_xor}")
    print(f"XOR Accuracy: {np.mean(xor_predictions == y_xor):.4f}")
    
    # Experiment 3: Universal Approximation Study
    print("\n" + "=" * 50)
    print("Experiment 3: Universal Approximation Study")
    print("=" * 50)
    
    # Generate non-linear function: y = sin(2πx) + 0.5*cos(4πx)
    x_approx = np.linspace(0, 1, 1000).reshape(-1, 1)
    y_approx = np.sin(2 * np.pi * x_approx) + 0.5 * np.cos(4 * np.pi * x_approx)
    
    # Split into train/test
    train_size = 800
    X_approx_train = x_approx[:train_size]
    y_approx_train = y_approx[:train_size]
    X_approx_test = x_approx[train_size:]
    y_approx_test = y_approx[train_size:]
    
    # Test different network sizes
    if QUICK_MODE:
        print("QUICK MODE: Testing only 2 network sizes (reduced from 5)")
        network_sizes = [
            ([16], "Medium (16 neurons)"),
            ([32, 32], "Deep (32-32 neurons)")
        ]
    elif HYBRID_MODE:
        print("HYBRID MODE: Testing 3 network sizes (reduced from 5)")
        network_sizes = [
            ([16], "Medium (16 neurons)"),
            ([64], "Large (64 neurons)"),
            ([32, 32], "Deep (32-32 neurons)")
        ]
    else:
        network_sizes = [
            ([4], "Small (4 neurons)"),
            ([16], "Medium (16 neurons)"),
            ([64], "Large (64 neurons)"),
            ([32, 32], "Deep (32-32 neurons)"),
            ([128, 64], "Very Large (128-64 neurons)")
        ]
    
    approximation_results = {}
    
    for hidden_sizes, name in network_sizes:
        print(f"\nTraining {name} network...")
        model_approx = MLP(
            input_size=1,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation='tanh'
        )
        
        # Train for function approximation (MSE loss)
        if QUICK_MODE:
            epochs_approx = 300
        elif HYBRID_MODE:
            epochs_approx = 500  # Reduced for faster execution (minimal quality mode)
        else:
            epochs_approx = 2000
        learning_rate_approx = 0.01
        
        for epoch in range(epochs_approx):
            # Forward pass
            y_pred = model_approx.forward(X_approx_train)
            
            # MSE loss and gradient
            mse_loss = np.mean((y_pred - y_approx_train) ** 2)
            grad_mse = 2 * (y_pred - y_approx_train) / len(X_approx_train)
            
            # Backward pass
            grads = model_approx.backward(grad_mse)
            model_approx.update(grads, learning_rate_approx)
            
            print_interval = 100 if QUICK_MODE else (200 if HYBRID_MODE else 500)
            if (epoch + 1) % print_interval == 0:
                test_pred = model_approx.forward(X_approx_test)
                test_mse = np.mean((test_pred - y_approx_test) ** 2)
                print(f"  Epoch {epoch+1}/{epochs_approx} - Train MSE: {mse_loss:.6f}, Test MSE: {test_mse:.6f}")
        
        # Final evaluation
        train_pred = model_approx.forward(X_approx_train)
        test_pred = model_approx.forward(X_approx_test)
        train_mse = np.mean((train_pred - y_approx_train) ** 2)
        test_mse = np.mean((test_pred - y_approx_test) ** 2)
        
        approximation_results[name] = {
            'model': model_approx,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_pred': train_pred,
            'test_pred': test_pred
        }
        
        print(f"  Final - Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
    
    # Visualize approximation results
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot 1: True function
    axes[0].plot(x_approx, y_approx, 'b-', linewidth=2, label='True Function')
    axes[0].set_title('True Function: sin(2πx) + 0.5cos(4πx)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot approximations for each network size
    for idx, (name, results) in enumerate(approximation_results.items(), 1):
        ax = axes[idx]
        ax.plot(x_approx, y_approx, 'b-', linewidth=2, label='True Function', alpha=0.5)
        ax.plot(X_approx_train, results['train_pred'], 'r--', linewidth=1.5, label='Approximation', alpha=0.8)
        ax.scatter(X_approx_train[::50], y_approx_train[::50], s=10, alpha=0.3, c='green')
        ax.set_title(f'{name}\nTest MSE: {results["test_mse"]:.6f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.legend()
    
    # Plot 6: MSE comparison
    ax = axes[5]
    names = list(approximation_results.keys())
    train_mses = [approximation_results[n]['train_mse'] for n in names]
    test_mses = [approximation_results[n]['test_mse'] for n in names]
    
    x_pos = np.arange(len(names))
    width = 0.35
    ax.bar(x_pos - width/2, train_mses, width, label='Train MSE', alpha=0.8)
    ax.bar(x_pos + width/2, test_mses, width, label='Test MSE', alpha=0.8)
    ax.set_xlabel('Network Size')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Approximation Quality vs Network Size')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.split('(')[0].strip() for n in names], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'section1', 'universal_approximation.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved universal approximation study to {save_path}")
    plt.close()
    
    print("\n" + "=" * 50)
    print("Section 1 experiments completed!")
    print("=" * 50)


if __name__ == '__main__':
    run_section1_experiments()
