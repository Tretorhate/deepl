"""
Optimizers implemented from scratch: SGD, Adam, RMSprop
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from section1.mlp import MLP, cross_entropy_loss, softmax
from utils.data_loader import load_mnist_numpy
from utils.visualization import plot_training_curves
from config import SECTION2_CONFIG, RESULTS_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)


class Optimizer:
    """Base class for optimizers."""
    
    def update(self, model, grads, learning_rate):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, momentum=0.0, weight_decay=0.0):
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = None
    
    def update(self, model, grads, learning_rate):
        """Update weights using SGD."""
        if self.velocity is None:
            self.velocity = [np.zeros_like(layer.weights) for layer in model.layers]
        
        for i, (layer, (grad_weights, grad_bias)) in enumerate(zip(model.layers, grads)):
            # Add weight decay
            if self.weight_decay > 0:
                grad_weights += self.weight_decay * layer.weights
            
            # Momentum update
            if self.momentum > 0:
                self.velocity[i] = self.momentum * self.velocity[i] - learning_rate * grad_weights
                layer.weights += self.velocity[i]
            else:
                layer.weights -= learning_rate * grad_weights
            
            layer.bias -= learning_rate * grad_bias


class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.t = 0     # Time step
    
    def update(self, model, grads, learning_rate):
        """Update weights using Adam."""
        self.t += 1
        
        if self.m is None:
            self.m = [np.zeros_like(layer.weights) for layer in model.layers]
            self.v = [np.zeros_like(layer.weights) for layer in model.layers]
        
        for i, (layer, (grad_weights, grad_bias)) in enumerate(zip(model.layers, grads)):
            # Add weight decay
            if self.weight_decay > 0:
                grad_weights += self.weight_decay * layer.weights
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad_weights
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad_weights ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update weights
            layer.weights -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            layer.bias -= learning_rate * grad_bias


class RMSprop(Optimizer):
    """RMSprop optimizer."""
    
    def __init__(self, alpha=0.99, epsilon=1e-8, weight_decay=0.0):
        self.alpha = alpha
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.cache = None
    
    def update(self, model, grads, learning_rate):
        """Update weights using RMSprop."""
        if self.cache is None:
            self.cache = [np.zeros_like(layer.weights) for layer in model.layers]
        
        for i, (layer, (grad_weights, grad_bias)) in enumerate(zip(model.layers, grads)):
            # Add weight decay
            if self.weight_decay > 0:
                grad_weights += self.weight_decay * layer.weights
            
            # Update cache (moving average of squared gradients)
            self.cache[i] = self.alpha * self.cache[i] + (1 - self.alpha) * (grad_weights ** 2)
            
            # Update weights
            layer.weights -= learning_rate * grad_weights / (np.sqrt(self.cache[i]) + self.epsilon)
            layer.bias -= learning_rate * grad_bias


def train_with_optimizer(X_train, y_train, X_val, y_val, model, optimizer, 
                         epochs, learning_rate, batch_size):
    """Train model with specified optimizer."""
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
            
            # Update with optimizer
            optimizer.update(model, grads, learning_rate)
            
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


def run_section2_experiments():
    """Run all Section 2 experiments."""
    from config import get_section_mode, is_quick_mode, is_hybrid_mode
    # Get mode for this section
    section_mode = get_section_mode(2)
    QUICK_MODE = is_quick_mode(2)
    HYBRID_MODE = is_hybrid_mode(2)
    
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist_numpy()
    
    # Adjust dataset size based on mode
    if QUICK_MODE:
        print("QUICK MODE: Using reduced dataset size")
        X_train = X_train[:10000]
        y_train = y_train[:10000]
        X_test = X_test[:2000]
        y_test = y_test[:2000]
    elif HYBRID_MODE:
        print("HYBRID MODE: Using moderate dataset size (25k samples)")
        X_train = X_train[:25000]
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
    print()
    
    # Experiment: Compare optimizers
    print("=" * 50)
    print("Experiment: Comparing Optimizers")
    print("=" * 50)
    
    # Use all optimizers in hybrid/full mode (required), fewer in quick mode
    if QUICK_MODE:
        print("QUICK MODE: Testing only Adam optimizer")
        optimizers_config = [('Adam', Adam())]
    else:
        optimizers_config = [
            ('SGD', SGD(momentum=0.9)),
            ('Adam', Adam()),
            ('RMSprop', RMSprop())
        ]
        if HYBRID_MODE:
            print("HYBRID MODE: Testing all 3 optimizers (SGD, Adam, RMSprop)")
    
    results = {}
    
    for opt_name, optimizer in optimizers_config:
        print(f"\nTraining with {opt_name} optimizer...")
        model = MLP(
            input_size=784,
            hidden_sizes=[128, 64],
            output_size=10,
            activation='relu'
        )
        
        train_losses, val_losses, train_accs, val_accs = train_with_optimizer(
            X_train, y_train, X_val, y_val,
            model, optimizer,
            SECTION2_CONFIG['epochs'],
            SECTION2_CONFIG['learning_rates'][0],
            SECTION2_CONFIG['batch_size']
        )
        
        results[opt_name] = {
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
    
    for opt_name in results.keys():
        epochs = range(1, len(results[opt_name]['val_losses']) + 1)
        ax1.plot(epochs, results[opt_name]['val_losses'], label=opt_name)
        ax2.plot(epochs, results[opt_name]['val_accs'], label=opt_name)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Optimizer Comparison - Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Optimizer Comparison - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'section2', 'optimizer_comparison.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved optimizer comparison to {save_path}")
    plt.close()
    
    # Experiment 2: Learning Rate Schedules
    print("\n" + "=" * 50)
    print("Experiment 2: Learning Rate Schedules")
    print("=" * 50)
    
    def step_decay(epoch, initial_lr=0.01, decay_rate=0.5, decay_step=10):
        """Step decay schedule."""
        return initial_lr * (decay_rate ** (epoch // decay_step))
    
    def exponential_decay(epoch, initial_lr=0.01, decay_rate=0.95):
        """Exponential decay schedule."""
        return initial_lr * (decay_rate ** epoch)
    
    def polynomial_decay(epoch, initial_lr=0.01, max_epochs=20, power=0.5):
        """Polynomial decay schedule."""
        return initial_lr * ((1 - epoch / max_epochs) ** power)
    
    # Use all schedules in hybrid/full mode, fewer in quick mode
    if QUICK_MODE:
        print("QUICK MODE: Testing only Constant and Step Decay schedules")
        schedules = {
            'Constant': lambda e, lr: lr,
            'Step Decay': step_decay
        }
    else:
        schedules = {
            'Constant': lambda e, lr: lr,
            'Step Decay': step_decay,
            'Exponential Decay': exponential_decay,
            'Polynomial Decay': polynomial_decay
        }
        if HYBRID_MODE:
            print("HYBRID MODE: Testing all 4 learning rate schedules")
    
    lr_results = {}
    initial_lr = 0.01
    
    for schedule_name, schedule_fn in schedules.items():
        print(f"\nTraining with {schedule_name} learning rate schedule...")
        model = MLP(
            input_size=784,
            hidden_sizes=[128, 64],
            output_size=10,
            activation='relu'
        )
        
        train_losses = []
        val_losses = []
        learning_rates_used = []
        
        n_train = X_train.shape[0]
        n_batches = n_train // SECTION2_CONFIG['batch_size']
        epochs = SECTION2_CONFIG['epochs']
        
        for epoch in range(epochs):
            lr = schedule_fn(epoch, initial_lr)
            learning_rates_used.append(lr)
            
            epoch_loss = 0
            correct = 0
            
            indices = np.random.permutation(n_train)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            optimizer = Adam()
            
            for i in range(n_batches):
                start_idx = i * SECTION2_CONFIG['batch_size']
                end_idx = min((i + 1) * SECTION2_CONFIG['batch_size'], n_train)
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                y_pred = model.forward(X_batch)
                loss, grad = cross_entropy_loss(y_pred, y_batch)
                grads = model.backward(grad)
                optimizer.update(model, grads, lr)
                
                epoch_loss += loss
                predictions = np.argmax(softmax(y_pred), axis=1)
                correct += np.sum(predictions == y_batch)
            
            train_loss = epoch_loss / n_batches
            train_acc = correct / n_train
            
            y_val_pred = model.forward(X_val)
            val_loss, _ = cross_entropy_loss(y_val_pred, y_val)
            val_predictions = np.argmax(softmax(y_val_pred), axis=1)
            val_acc = np.mean(val_predictions == y_val)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        
        lr_results[schedule_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': learning_rates_used
        }
    
    # Plot learning rate schedules
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Learning rate schedules
    ax = axes[0, 0]
    for name, results in lr_results.items():
        epochs_range = range(len(results['learning_rates']))
        ax.plot(epochs_range, results['learning_rates'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedules')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')
    
    # Plot 2: Validation loss comparison
    ax = axes[0, 1]
    for name, results in lr_results.items():
        epochs_range = range(len(results['val_losses']))
        ax.plot(epochs_range, results['val_losses'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss with Different LR Schedules')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Training loss comparison
    ax = axes[1, 0]
    for name, results in lr_results.items():
        epochs_range = range(len(results['train_losses']))
        ax.plot(epochs_range, results['train_losses'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss with Different LR Schedules')
    ax.legend()
    ax.grid(True)
    
    # Plot 4: Final validation loss comparison
    ax = axes[1, 1]
    names = list(lr_results.keys())
    final_val_losses = [lr_results[n]['val_losses'][-1] for n in names]
    ax.bar(names, final_val_losses, alpha=0.7)
    ax.set_ylabel('Final Validation Loss')
    ax.set_title('Final Performance Comparison')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'section2', 'learning_rate_schedules.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved learning rate schedule analysis to {save_path}")
    plt.close()
    
    # Experiment 3: Gradient Flow Analysis
    print("\n" + "=" * 50)
    print("Experiment 3: Gradient Flow and Vanishing Gradient Analysis")
    print("=" * 50)
    
    # Create deep network to analyze gradient flow
    if QUICK_MODE:
        print("QUICK MODE: Testing only 2 network depths")
        deep_networks = [
            ([64], "Shallow (1 layer)"),
            ([64, 64, 64], "Deep (3 layers)")
        ]
    elif HYBRID_MODE:
        print("HYBRID MODE: Testing 3 network depths")
        deep_networks = [
            ([64], "Shallow (1 layer)"),
            ([64, 64], "Medium (2 layers)"),
            ([64, 64, 64], "Deep (3 layers)")
        ]
    else:
        deep_networks = [
            ([64], "Shallow (1 layer)"),
            ([64, 64], "Medium (2 layers)"),
            ([64, 64, 64], "Deep (3 layers)"),
            ([64, 64, 64, 64], "Very Deep (4 layers)")
        ]
    
    gradient_analysis = {}
    
    for hidden_sizes, name in deep_networks:
        print(f"\nAnalyzing gradient flow for {name}...")
        model = MLP(
            input_size=784,
            hidden_sizes=hidden_sizes,
            output_size=10,
            activation='sigmoid'  # Use sigmoid to demonstrate vanishing gradients
        )
        
        # Get a small batch
        X_sample = X_train[:32]
        y_sample = y_train[:32]
        
        # Forward pass
        y_pred = model.forward(X_sample)
        loss, grad = cross_entropy_loss(y_pred, y_sample)
        
        # Backward pass
        grads = model.backward(grad)
        
        # Analyze gradient magnitudes
        grad_magnitudes = []
        for layer_idx, (layer, (grad_weights, grad_bias)) in enumerate(zip(model.layers, grads)):
            grad_norm = np.linalg.norm(grad_weights)
            grad_magnitudes.append(grad_norm)
            print(f"  Layer {layer_idx + 1}: Gradient norm = {grad_norm:.6f}")
        
        gradient_analysis[name] = {
            'grad_magnitudes': grad_magnitudes,
            'num_layers': len(hidden_sizes) + 1
        }
    
    # Visualize gradient flow
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Gradient magnitudes by layer
    ax = axes[0]
    for name, analysis in gradient_analysis.items():
        layers = range(1, analysis['num_layers'] + 1)
        ax.plot(layers, analysis['grad_magnitudes'], marker='o', label=name, linewidth=2)
    ax.set_xlabel('Layer Number')
    ax.set_ylabel('Gradient Magnitude (L2 norm)')
    ax.set_title('Gradient Flow Through Network Layers')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')
    
    # Plot 2: Gradient ratio (first to last layer)
    ax = axes[1]
    names = list(gradient_analysis.keys())
    ratios = []
    for name in names:
        grads = gradient_analysis[name]['grad_magnitudes']
        if len(grads) > 1:
            ratio = grads[0] / (grads[-1] + 1e-10)
            ratios.append(ratio)
        else:
            ratios.append(1.0)
    
    ax.bar(names, ratios, alpha=0.7)
    ax.set_ylabel('Gradient Ratio (First/Last Layer)')
    ax.set_title('Vanishing Gradient Analysis')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'section2', 'gradient_flow_analysis.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved gradient flow analysis to {save_path}")
    plt.close()
    
    print("\n" + "=" * 50)
    print("Section 2 experiments completed!")
    print("=" * 50)


if __name__ == '__main__':
    run_section2_experiments()
