"""
Complete CNN implementation from scratch for MNIST classification.
Includes full backpropagation and training pipeline.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from section3.cnn_layers import Conv2D, MaxPool2D, AvgPool2D
from section1.mlp import LinearLayer, cross_entropy_loss, softmax
from utils.data_loader import load_mnist_numpy
from utils.visualization import plot_training_curves, plot_confusion_matrix
from config import SECTION3_CONFIG, RESULTS_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)


class Conv2DComplete(Conv2D):
    """Complete Conv2D with full backward pass."""
    
    def backward(self, grad_output):
        """Optimized backward pass using vectorized operations."""
        batch_size, in_channels, in_height, in_width = self.input.shape
        out_channels, _, kernel_h, kernel_w = self.weights.shape
        out_height, out_width = grad_output.shape[2], grad_output.shape[3]
        
        # Gradient w.r.t. bias (vectorized - sum over batch, height, width)
        grad_bias = np.sum(grad_output, axis=(0, 2, 3))
        
        # Add padding to input if needed
        if self.padding > 0:
            input_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), 
                                             (self.padding, self.padding)), mode='constant')
            grad_input_padded = np.zeros_like(input_padded)
            padded_h, padded_w = input_padded.shape[2], input_padded.shape[3]
        else:
            input_padded = self.input
            grad_input_padded = np.zeros_like(self.input)
            padded_h, padded_w = in_height, in_width
        
        # Optimized gradient computation using vectorized operations
        grad_weights = np.zeros_like(self.weights)
        
        # Vectorized computation for grad_weights (reduced from 4 nested loops to 2)
        for oh in range(out_height):
            for ow in range(out_width):
                h_start = oh * self.stride
                w_start = ow * self.stride
                h_end = h_start + kernel_h
                w_end = w_start + kernel_w
                
                # Extract input patch for all batches and channels: (B, C, H, W)
                input_patch = input_padded[:, :, h_start:h_end, w_start:w_end]  # (B, C, kernel_h, kernel_w)
                # Extract grad_output for all batches and output channels: (B, OC)
                grad_patch = grad_output[:, :, oh, ow]  # (B, OC)
                
                # Vectorized: grad_weights[oc, ic, kh, kw] = sum_b(input_patch[b, ic, kh, kw] * grad_patch[b, oc])
                # Using einsum: sum over batch dimension, keep spatial dimensions
                # 'bihw' = batch, input_channel, height, width (4 dims)
                # 'bo' = batch, output_channel (2 dims)
                # 'oihw' = output_channel, input_channel, height, width (4 dims)
                grad_weights += np.einsum('bihw,bo->oihw', input_patch, grad_patch)
        
        # Vectorized computation for grad_input
        for oc in range(out_channels):
            for oh in range(out_height):
                for ow in range(out_width):
                    h_start = oh * self.stride
                    w_start = ow * self.stride
                    h_end = h_start + kernel_h
                    w_end = w_start + kernel_w
                    
                    # Vectorized: add weights * grad_output for all batches
                    grad_input_padded[:, :, h_start:h_end, w_start:w_end] += \
                        self.weights[oc, :, :, :][None, :, :, :] * grad_output[:, oc, oh, ow][:, None, None, None]
        
        # Remove padding from grad_input
        if self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_padded
        
        return grad_input, grad_weights, grad_bias


class Flatten:
    """Flatten layer to convert 2D feature maps to 1D."""
    
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)


class CNN:
    """Complete CNN for image classification."""
    
    def __init__(self, num_classes=10, use_maxpool=True):
        """
        Initialize CNN.
        
        Args:
            num_classes: Number of output classes
            use_maxpool: If True, use MaxPool, else use AvgPool
        """
        self.layers = []
        self.use_maxpool = use_maxpool
        
        # Conv layers
        self.conv1 = Conv2DComplete(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool2D(pool_size=2, stride=2) if use_maxpool else AvgPool2D(pool_size=2, stride=2)
        
        self.conv2 = Conv2DComplete(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = MaxPool2D(pool_size=2, stride=2) if use_maxpool else AvgPool2D(pool_size=2, stride=2)
        
        # Flatten
        self.flatten = Flatten()
        
        # Fully connected layers
        self.fc1 = LinearLayer(64 * 7 * 7, 128)  # After 2 pooling layers: 28->14->7
        self.fc2 = LinearLayer(128, num_classes)
        
        # Store activations for backward pass
        self.activations = []
    
    def forward(self, x):
        """Forward pass."""
        self.activations = []
        
        # Conv block 1
        x = self.conv1.forward(x)
        self.activations.append(('conv1', x))
        x = np.maximum(0, x)  # ReLU
        x = self.pool1.forward(x)
        self.activations.append(('pool1', x))
        
        # Conv block 2
        x = self.conv2.forward(x)
        self.activations.append(('conv2', x))
        x = np.maximum(0, x)  # ReLU
        x = self.pool2.forward(x)
        self.activations.append(('pool2', x))
        
        # Flatten
        x = self.flatten.forward(x)
        
        # Fully connected
        x = self.fc1.forward(x)
        self.fc1_output = x.copy()  # Store fc1 output before ReLU for backward pass
        x = np.maximum(0, x)  # ReLU
        x = self.fc2.forward(x)
        
        return x
    
    def backward(self, grad_output):
        """Backward pass."""
        # FC layers backward
        grad_fc2_input, grad_fc2_weights, grad_fc2_bias = self.fc2.backward(grad_output)
        # Apply ReLU derivative: gradient w.r.t. fc1 output (before ReLU)
        grad_fc1_output = grad_fc2_input * (self.fc1_output > 0).astype(float)  # ReLU derivative
        
        grad_fc1_input, grad_fc1_weights, grad_fc1_bias = self.fc1.backward(grad_fc1_output)
        # No ReLU before fc1, so no derivative needed here
        
        # Flatten backward
        grad_flat = self.flatten.backward(grad_fc1_input)
        
        # Pool and Conv backward
        grad_pool2 = self.pool2.backward(grad_flat)
        grad_pool2 = grad_pool2 * (self.activations[2][1] > 0).astype(float)  # ReLU derivative
        
        grad_conv2_input, grad_conv2_weights, grad_conv2_bias = self.conv2.backward(grad_pool2)
        
        grad_pool1 = self.pool1.backward(grad_conv2_input)
        grad_pool1 = grad_pool1 * (self.activations[0][1] > 0).astype(float)  # ReLU derivative
        
        grad_conv1_input, grad_conv1_weights, grad_conv1_bias = self.conv1.backward(grad_pool1)
        
        return [
            (grad_conv1_weights, grad_conv1_bias),
            (grad_conv2_weights, grad_conv2_bias),
            (grad_fc1_weights, grad_fc1_bias),
            (grad_fc2_weights, grad_fc2_bias)
        ]
    
    def update(self, grads, learning_rate):
        """Update weights."""
        (grad_conv1_weights, grad_conv1_bias), \
        (grad_conv2_weights, grad_conv2_bias), \
        (grad_fc1_weights, grad_fc1_bias), \
        (grad_fc2_weights, grad_fc2_bias) = grads
        
        self.conv1.weights -= learning_rate * grad_conv1_weights
        self.conv1.bias -= learning_rate * grad_conv1_bias
        self.conv2.weights -= learning_rate * grad_conv2_weights
        self.conv2.bias -= learning_rate * grad_conv2_bias
        
        self.fc1.update(grad_fc1_weights, grad_fc1_bias, learning_rate)
        self.fc2.update(grad_fc2_weights, grad_fc2_bias, learning_rate)


def train_cnn(X_train, y_train, X_val, y_val, model, epochs, learning_rate, batch_size):
    """Train CNN model."""
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    n_train = X_train.shape[0]
    n_batches = n_train // batch_size
    
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        
        indices = np.random.permutation(n_train)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        print(f"Epoch {epoch+1}/{epochs} - Processing {n_batches} batches...")
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
            
            # Progress printing every 25 batches
            if (i + 1) % 25 == 0 or (i + 1) == n_batches:
                print(f"  Batch {i+1}/{n_batches} - Loss: {loss:.4f}")
        
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
        
        # Always print epoch results
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")
        print()
    
    return train_losses, val_losses, train_accs, val_accs


def calculate_receptive_field(layers):
    """
    Calculate theoretical receptive field for CNN.
    
    Args:
        layers: List of layer configurations [(type, kernel_size, stride), ...]
    
    Returns:
        receptive_field: Theoretical receptive field size
    """
    rf = 1
    jump = 1
    
    for layer_type, kernel_size, stride in layers:
        if layer_type == 'conv':
            rf += (kernel_size - 1) * jump
            jump *= stride
        elif layer_type == 'pool':
            rf += (kernel_size - 1) * jump
            jump *= stride
    
    return rf


def run_cnn_training():
    """Train complete CNN on MNIST."""
    from config import get_section_mode, is_quick_mode, is_hybrid_mode
    # Get mode for this section
    section_mode = get_section_mode(3)
    QUICK_MODE = is_quick_mode(3)
    HYBRID_MODE = is_hybrid_mode(3)
    
    print("=" * 50)
    print("Complete CNN Training on MNIST")
    print("=" * 50)
    print()
    
    # Load and prepare MNIST data
    print("Loading MNIST dataset...")
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
    
    # Reshape to (N, 1, 28, 28) for CNN
    X_train = X_train.reshape(-1, 1, 28, 28) / 255.0
    X_test = X_test.reshape(-1, 1, 28, 28) / 255.0
    
    # Split training into train/val
    if QUICK_MODE:
        val_size = min(1000, len(X_train) // 5)
    elif HYBRID_MODE:
        val_size = 1000  # Reduced validation size
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
    
    # Train CNN with MaxPool
    print("Training CNN with MaxPool...")
    model_maxpool = CNN(num_classes=10, use_maxpool=True)
    
    if QUICK_MODE:
        epochs_cnn = 3
        batch_size_cnn = 64
    elif HYBRID_MODE:
        epochs_cnn = 3  # Reduced for faster execution (minimal quality mode)
        batch_size_cnn = 32
    else:
        epochs_cnn = 10
        batch_size_cnn = 32
    train_losses, val_losses, train_accs, val_accs = train_cnn(
        X_train, y_train, X_val, y_val,
        model_maxpool,
        epochs=epochs_cnn,
        learning_rate=0.001,
        batch_size=batch_size_cnn
    )
    
    # Test accuracy (batched to avoid memory issues)
    test_batch_size = 1000  # Process test set in batches
    test_predictions = []
    print("Evaluating on test set (batched)...")
    for i in range(0, len(X_test), test_batch_size):
        X_test_batch = X_test[i:i+test_batch_size]
        y_test_batch = y_test[i:i+test_batch_size]
        y_test_pred_batch = model_maxpool.forward(X_test_batch)
        test_predictions_batch = np.argmax(softmax(y_test_pred_batch), axis=1)
        test_predictions.append(test_predictions_batch)
    test_predictions = np.concatenate(test_predictions)
    test_acc = np.mean(test_predictions == y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    # Plot training curves
    save_path = os.path.join(RESULTS_DIR, 'section3', 'cnn_training_curves.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                        save_path=save_path, title="CNN Training on MNIST")
    
    # Confusion matrix
    save_path = os.path.join(RESULTS_DIR, 'section3', 'cnn_confusion_matrix.png')
    plot_confusion_matrix(y_test, test_predictions,
                         class_names=[str(i) for i in range(10)],
                         save_path=save_path, title="CNN Confusion Matrix")
    
    return model_maxpool, test_acc


if __name__ == '__main__':
    run_cnn_training()
