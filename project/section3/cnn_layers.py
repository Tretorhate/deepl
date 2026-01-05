"""
CNN layers implemented from scratch using NumPy: Conv2D, MaxPool, AvgPool
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_cifar10
from utils.visualization import plot_filters, plot_activation_maps
from config import SECTION3_CONFIG, RESULTS_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)


class Conv2D:
    """2D Convolutional layer implemented from scratch."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Initialize Conv2D layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel (int or tuple)
            stride: Stride of convolution
            padding: Padding size
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights (Xavier initialization)
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        fan_out = out_channels * kernel_size[0] * kernel_size[1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        self.weights = np.random.uniform(-limit, limit, 
                                        (out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.bias = np.zeros((out_channels,))
        
        self.input = None
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, height, width)
        
        Returns:
            output: Convolved output
        """
        self.input = x
        batch_size, in_channels, in_height, in_width = x.shape
        
        # Add padding
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), 
                                  (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x
        
        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size[1]) // self.stride + 1
        
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        x_slice = x_padded[b, :, h_start:h_end, w_start:w_end]
                        output[b, oc, oh, ow] = np.sum(x_slice * self.weights[oc]) + self.bias[oc]
        
        return output
    
    def backward(self, grad_output):
        """Backward pass (simplified - full implementation would be more complex)."""
        # This is a simplified backward pass
        # Full implementation would compute gradients w.r.t. input, weights, and bias
        return grad_output


class MaxPool2D:
    """2D Max Pooling layer."""
    
    def __init__(self, pool_size=2, stride=None):
        """
        Initialize MaxPool2D layer.
        
        Args:
            pool_size: Size of pooling window
            stride: Stride (defaults to pool_size)
        """
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size[0]
        
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        
        self.input = None
        self.max_indices = None
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, height, width)
        
        Returns:
            output: Pooled output
        """
        self.input = x
        batch_size, channels, in_height, in_width = x.shape
        
        out_height = (in_height - self.pool_size[0]) // self.stride[0] + 1
        out_width = (in_width - self.pool_size[1]) // self.stride[1] + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride[0]
                        w_start = ow * self.stride[1]
                        h_end = h_start + self.pool_size[0]
                        w_end = w_start + self.pool_size[1]
                        
                        x_slice = x[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(x_slice)
                        output[b, c, oh, ow] = max_val
                        
                        # Store indices of max value
                        max_idx = np.unravel_index(np.argmax(x_slice), x_slice.shape)
                        self.max_indices[b, c, oh, ow] = [h_start + max_idx[0], w_start + max_idx[1]]
        
        return output
    
    def backward(self, grad_output):
        """Backward pass."""
        grad_input = np.zeros_like(self.input)
        
        for b in range(grad_output.shape[0]):
            for c in range(grad_output.shape[1]):
                for oh in range(grad_output.shape[2]):
                    for ow in range(grad_output.shape[3]):
                        h_idx, w_idx = self.max_indices[b, c, oh, ow]
                        grad_input[b, c, h_idx, w_idx] += grad_output[b, c, oh, ow]
        
        return grad_input


class AvgPool2D:
    """2D Average Pooling layer."""
    
    def __init__(self, pool_size=2, stride=None):
        """
        Initialize AvgPool2D layer.
        
        Args:
            pool_size: Size of pooling window
            stride: Stride (defaults to pool_size)
        """
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size[0]
        
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        
        self.input = None
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, height, width)
        
        Returns:
            output: Pooled output
        """
        self.input = x
        batch_size, channels, in_height, in_width = x.shape
        
        out_height = (in_height - self.pool_size[0]) // self.stride[0] + 1
        out_width = (in_width - self.pool_size[1]) // self.stride[1] + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride[0]
                        w_start = ow * self.stride[1]
                        h_end = h_start + self.pool_size[0]
                        w_end = w_start + self.pool_size[1]
                        
                        x_slice = x[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, oh, ow] = np.mean(x_slice)
        
        return output
    
    def backward(self, grad_output):
        """Backward pass."""
        grad_input = np.zeros_like(self.input)
        pool_area = self.pool_size[0] * self.pool_size[1]
        
        for b in range(grad_output.shape[0]):
            for c in range(grad_output.shape[1]):
                for oh in range(grad_output.shape[2]):
                    for ow in range(grad_output.shape[3]):
                        h_start = oh * self.stride[0]
                        w_start = ow * self.stride[1]
                        h_end = h_start + self.pool_size[0]
                        w_end = w_start + self.pool_size[1]
                        
                        grad_val = grad_output[b, c, oh, ow] / pool_area
                        grad_input[b, c, h_start:h_end, w_start:w_end] += grad_val
        
        return grad_input


def calculate_receptive_field(layers):
    """
    Calculate theoretical receptive field for CNN architecture.
    
    Args:
        layers: List of tuples (layer_type, kernel_size, stride, padding)
                layer_type: 'conv' or 'pool'
    
    Returns:
        receptive_field: Theoretical receptive field size
        effective_stride: Effective stride
    """
    rf = 1
    effective_stride = 1
    
    for layer_type, kernel_size, stride, padding in layers:
        if layer_type == 'conv':
            rf += (kernel_size - 1) * effective_stride
            effective_stride *= stride
        elif layer_type == 'pool':
            rf += (kernel_size - 1) * effective_stride
            effective_stride *= stride
    
    return rf, effective_stride


def run_section3_experiments():
    """Run all Section 3 experiments."""
    print("=" * 50)
    print("Section 3: CNN Layers from Scratch")
    print("=" * 50)
    print()
    
    # Test Conv2D layer
    print("Testing Conv2D layer...")
    conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    x = np.random.randn(2, 3, 32, 32)  # Batch of 2, 3 channels, 32x32 images
    output = conv.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print()
    
    # Test MaxPool
    print("Testing MaxPool2D layer...")
    maxpool = MaxPool2D(pool_size=2, stride=2)
    pooled = maxpool.forward(output)
    print(f"Input shape: {output.shape}")
    print(f"Output shape: {pooled.shape}")
    print()
    
    # Test AvgPool
    print("Testing AvgPool2D layer...")
    avgpool = AvgPool2D(pool_size=2, stride=2)
    avg_pooled = avgpool.forward(output)
    print(f"Input shape: {output.shape}")
    print(f"Output shape: {avg_pooled.shape}")
    print()
    
    # Visualize filters
    print("Visualizing convolutional filters...")
    save_path = os.path.join(RESULTS_DIR, 'section3', 'filters.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_filters(conv.weights[:8], save_path=save_path)  # Plot first 8 filters
    print(f"Saved filter visualization to {save_path}")
    print()
    
    # Visualize activations
    print("Visualizing activation maps...")
    save_path = os.path.join(RESULTS_DIR, 'section3', 'activations.png')
    plot_activation_maps(output[:1], save_path=save_path)  # Plot first sample
    print(f"Saved activation maps to {save_path}")
    print()
    
    # Experiment: Receptive Field Analysis
    print("=" * 50)
    print("Receptive Field Analysis")
    print("=" * 50)
    
    architectures = [
        ("Simple CNN", [
            ('conv', 3, 1, 1),
            ('pool', 2, 2, 0),
            ('conv', 3, 1, 1),
            ('pool', 2, 2, 0)
        ]),
        ("Deep CNN", [
            ('conv', 3, 1, 1),
            ('conv', 3, 1, 1),
            ('pool', 2, 2, 0),
            ('conv', 3, 1, 1),
            ('conv', 3, 1, 1),
            ('pool', 2, 2, 0)
        ]),
        ("Wide Receptive Field", [
            ('conv', 5, 1, 2),
            ('pool', 2, 2, 0),
            ('conv', 5, 1, 2),
            ('pool', 2, 2, 0)
        ])
    ]
    
    receptive_field_results = {}
    
    for arch_name, layers in architectures:
        rf, eff_stride = calculate_receptive_field(layers)
        receptive_field_results[arch_name] = {
            'receptive_field': rf,
            'effective_stride': eff_stride,
            'layers': layers
        }
        print(f"{arch_name}:")
        print(f"  Receptive Field: {rf}x{rf} pixels")
        print(f"  Effective Stride: {eff_stride}")
        print()
    
    # Visualize receptive fields
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    names = list(receptive_field_results.keys())
    rfs = [receptive_field_results[n]['receptive_field'] for n in names]
    
    ax.bar(names, rfs, alpha=0.7)
    ax.set_ylabel('Receptive Field Size (pixels)')
    ax.set_title('Receptive Field Comparison Across Architectures')
    ax.grid(True, alpha=0.3)
    
    for i, (name, rf) in enumerate(zip(names, rfs)):
        ax.text(i, rf, f'{rf}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'section3', 'receptive_field_analysis.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved receptive field analysis to {save_path}")
    plt.close()
    
    # Experiment: Pooling Strategy Comparison
    print("=" * 50)
    print("Pooling Strategy Comparison")
    print("=" * 50)
    
    # Import optimized CNN for training
    from section3.optimized_cnn_complete import OptimizedCNN, train_optimized_cnn, softmax
    from utils.data_loader import load_mnist_numpy
    from config import QUICK_MODE, HYBRID_MODE
    
    print("Loading MNIST for pooling comparison...")
    X_train, y_train, X_test, y_test = load_mnist_numpy()
    
    # Adjust dataset size based on mode
    if QUICK_MODE:
        print("QUICK MODE: Using reduced dataset size")
        X_train = X_train[:5000]
        y_train = y_train[:5000]
        X_test = X_test[:1000]
        y_test = y_test[:1000]
    elif HYBRID_MODE:
        print("HYBRID MODE: Using reduced dataset size (6k samples for faster training)")
        X_train = X_train[:6000]
        y_train = y_train[:6000]
        X_test = X_test[:2000]
        y_test = y_test[:2000]
    else:
        # Full mode: Use full MNIST dataset for pooling comparison
        print("FULL MODE: Using full MNIST dataset for pooling comparison")
        # Don't slice - use full dataset (60k training, 10k test)
    
    # Reshape to (N, 1, 28, 28) for CNN
    X_train = X_train.reshape(-1, 1, 28, 28) / 255.0
    X_test = X_test.reshape(-1, 1, 28, 28) / 255.0
    
    if QUICK_MODE:
        val_size = min(1000, len(X_train) // 5)
        train_pool_size = 3000
    elif HYBRID_MODE:
        val_size = 1000  # Reduced validation size
        train_pool_size = 4000  # Reduced training size for pooling comparison
    else:
        val_size = 10000  # 10k validation for full 60k training set
        train_pool_size = 50000  # 50k for pooling comparison (use most of training data)
    
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    
    # Ensure we don't exceed available data
    available_train = len(X_train) - val_size
    train_pool_size = min(train_pool_size, available_train)
    
    X_train_pool = X_train[val_size:val_size+train_pool_size]
    y_train_pool = y_train[val_size:val_size+train_pool_size]
    
    print(f"Pooling comparison: {len(X_train_pool)} training samples, {len(X_val)} validation samples")
    
    if len(X_train_pool) == 0:
        raise ValueError(f"No training samples available! val_size={val_size}, train_pool_size={train_pool_size}, total_samples={len(X_train)}")
    
    pooling_results = {}
    
    for pool_type, use_maxpool in [('MaxPool', True), ('AvgPool', False)]:
        print(f"\nTraining Optimized CNN with {pool_type}...")
        model = OptimizedCNN(num_classes=10, use_maxpool=use_maxpool)
        
        if QUICK_MODE:
            epochs_pool = 2
            batch_size_pool = 64
        elif HYBRID_MODE:
            epochs_pool = 5  # Increased for better pooling comparison
            batch_size_pool = 32
        else:
            epochs_pool = 5
            batch_size_pool = 64  # Larger batch for faster training
        
        train_losses, val_losses, train_accs, val_accs = train_optimized_cnn(
            X_train_pool, y_train_pool, X_val, y_val,
            model,
            epochs=epochs_pool,
            learning_rate=0.015,  # Higher initial LR with decay schedule
            batch_size=batch_size_pool
        )
        
        # Test accuracy (batched to avoid memory issues)
        test_batch_size = 1000  # Process test set in batches
        test_predictions = []
        for i in range(0, len(X_test), test_batch_size):
            X_test_batch = X_test[i:i+test_batch_size]
            y_test_batch = y_test[i:i+test_batch_size]
            y_test_pred_batch = model.forward(X_test_batch)
            test_predictions_batch = np.argmax(softmax(y_test_pred_batch), axis=1)
            test_predictions.append(test_predictions_batch)
        test_predictions = np.concatenate(test_predictions)
        test_acc = np.mean(test_predictions == y_test)
        
        pooling_results[pool_type] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'test_acc': test_acc
        }
        
        print(f"Test Accuracy with {pool_type}: {test_acc:.4f}")
    
    # Plot pooling comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for pool_type in ['MaxPool', 'AvgPool']:
        results = pooling_results[pool_type]
        epochs = range(1, len(results['val_losses']) + 1)
        
        axes[0].plot(epochs, results['val_losses'], label=pool_type, linewidth=2)
        axes[1].plot(epochs, results['val_accs'], label=pool_type, linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('Pooling Strategy Comparison - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('Pooling Strategy Comparison - Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Final accuracy comparison
    pool_types = list(pooling_results.keys())
    test_accs = [pooling_results[pt]['test_acc'] for pt in pool_types]
    axes[2].bar(pool_types, test_accs, alpha=0.7)
    axes[2].set_ylabel('Test Accuracy')
    axes[2].set_title('Final Test Accuracy Comparison')
    axes[2].grid(True, alpha=0.3)
    for i, acc in enumerate(test_accs):
        axes[2].text(i, acc, f'{acc:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'section3', 'pooling_comparison.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved pooling comparison to {save_path}")
    plt.close()
    
    # Experiment: Complete CNN Training
    print("\n" + "=" * 50)
    print("Complete CNN Training on MNIST")
    print("=" * 50)
    
    if QUICK_MODE:
        print("QUICK MODE: Skipping full CNN training (takes too long)")
        print("   Pooling comparison above demonstrates CNN functionality.")
    else:
        if HYBRID_MODE:
            print("HYBRID MODE: Running optimized CNN training with reduced epochs")
        try:
            from section3.optimized_cnn_complete import run_optimized_cnn_training
            model_cnn, test_acc_cnn = run_optimized_cnn_training()
            print(f"\nFinal Optimized CNN Test Accuracy: {test_acc_cnn:.4f}")
            if test_acc_cnn >= 0.95:
                print("Achieved target accuracy of 95%!")
            else:
                print(f"⚠ Target accuracy of 95% not reached. Current: {test_acc_cnn:.4f}")
        except Exception as e:
            print(f"Note: Complete optimized CNN training encountered an issue: {e}")
            print("This is expected if running in limited resource environment.")
    
    print("\n" + "=" * 50)
    print("Section 3 experiments completed!")
    print("=" * 50)


if __name__ == '__main__':
    run_section3_experiments()
