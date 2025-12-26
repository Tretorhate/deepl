"""
Problem 2: Optimization and Training Dynamics
Parts A, B, C - Optimizers, Gradients, Regularization
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {device}')


class MLP(nn.Module):
    """Basic MLP"""
    def __init__(self, layer_sizes, activation='relu', dropout=0.0):
        super().__init__()
        
        acts = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}
        self.act = acts[activation]
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(self.act)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class DeepMLP(nn.Module):
    """Deep MLP for gradient analysis - tracks gradients per layer"""
    def __init__(self, layer_sizes, activation='relu'):
        super().__init__()
        
        acts = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.activations.append(acts[activation]())
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.activations):
                x = self.activations[i](x)
        return x


def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    return DataLoader(train_data, batch_size=64, shuffle=True), DataLoader(test_data, batch_size=1000)


def train_with_optimizer(model, train_loader, test_loader, optimizer, epochs=20):
    """Train with given optimizer, return history and time"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'test_acc': []}
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        train_loss = total_loss / len(train_loader)
        test_acc = 100. * correct / total
        history['train_loss'].append(train_loss)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | Test: {test_acc:.2f}%')
    
    return history, time.time() - start


def train_with_gradients(model, train_loader, epochs=10):
    """Train and track gradient magnitudes at each layer"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    grad_history = {i: [] for i in range(len(model.layers))}
    
    for epoch in range(epochs):
        model.train()
        epoch_grads = {i: [] for i in range(len(model.layers))}
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            
            # grab gradient magnitudes
            for i, layer in enumerate(model.layers):
                if layer.weight.grad is not None:
                    grad_norm = layer.weight.grad.norm().item()
                    epoch_grads[i].append(grad_norm)
            
            optimizer.step()
        
        # avg gradients for this epoch
        for i in epoch_grads:
            if epoch_grads[i]:
                grad_history[i].append(np.mean(epoch_grads[i]))
        
        print(f'Epoch {epoch+1}/{epochs} done')
    
    return grad_history


def part_a_optimizers():
    """Compare different optimizers"""
    print('\n' + '='*50)
    print('PART A: Optimizer Comparison')
    print('='*50)
    
    train_loader, test_loader = load_mnist()
    layer_sizes = [784, 128, 64, 10]
    
    optimizers_config = {
        'SGD': lambda p: optim.SGD(p, lr=0.01),
        'SGD+Momentum': lambda p: optim.SGD(p, lr=0.01, momentum=0.9),
        'RMSprop': lambda p: optim.RMSprop(p, lr=0.001),
        'Adam': lambda p: optim.Adam(p, lr=0.001)
    }
    
    results = {}
    times = {}
    
    for name, opt_fn in optimizers_config.items():
        print(f'\n--- {name} ---')
        model = MLP(layer_sizes, activation='relu')
        optimizer = opt_fn(model.parameters())
        history, elapsed = train_with_optimizer(model, train_loader, test_loader, optimizer, epochs=20)
        results[name] = history
        times[name] = elapsed
    
    # plot loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name in results:
        plt.plot(results[name]['train_loss'], label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for name in results:
        plt.plot(results[name]['test_acc'], label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/part2a_optimizer_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # summary table
    print('\n' + '='*55)
    print(f'{"Optimizer":<15} {"Test Accuracy":<15} {"Time (s)":<10}')
    print('-'*40)
    for name in results:
        print(f'{name:<15} {results[name]["test_acc"][-1]:.2f}%          {times[name]:.1f}')
    print('='*55)
    
    return results


def part_b_gradients():
    """Analyze vanishing gradients"""
    print('\n' + '='*50)
    print('PART B: Gradient Analysis')
    print('='*50)
    
    train_loader, _ = load_mnist()
    
    # deep network: 6 hidden layers
    layer_sizes = [784, 256, 128, 64, 32, 16, 10]
    
    print(f'\nDeep network: {layer_sizes}')
    
    # sigmoid
    print('\n--- Training with SIGMOID ---')
    model_sig = DeepMLP(layer_sizes, activation='sigmoid')
    grads_sigmoid = train_with_gradients(model_sig, train_loader, epochs=10)
    
    # relu
    print('\n--- Training with RELU ---')
    model_relu = DeepMLP(layer_sizes, activation='relu')
    grads_relu = train_with_gradients(model_relu, train_loader, epochs=10)
    
    # plot gradient magnitudes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i in range(len(layer_sizes) - 1):
        axes[0].plot(grads_sigmoid[i], label=f'Layer {i+1}', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Gradient Magnitude')
    axes[0].set_title('Sigmoid - Gradient Magnitudes per Layer')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    for i in range(len(layer_sizes) - 1):
        axes[1].plot(grads_relu[i], label=f'Layer {i+1}', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Gradient Magnitude')
    axes[1].set_title('ReLU - Gradient Magnitudes per Layer')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/part2b_gradient_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # avg gradient per layer (final epoch)
    print('\n=== Final Epoch Gradient Magnitudes ===')
    print(f'{"Layer":<10} {"Sigmoid":<15} {"ReLU":<15}')
    print('-'*40)
    for i in range(len(layer_sizes) - 1):
        sig_val = grads_sigmoid[i][-1] if grads_sigmoid[i] else 0
        relu_val = grads_relu[i][-1] if grads_relu[i] else 0
        print(f'Layer {i+1:<4} {sig_val:.6f}        {relu_val:.6f}')


def part_c_regularization():
    """Compare regularization techniques"""
    print('\n' + '='*50)
    print('PART C: Regularization Effects')
    print('='*50)
    
    train_loader, test_loader = load_mnist()
    layer_sizes = [784, 256, 128, 10]  # slightly bigger to show overfitting
    
    results = {}
    
    # 1. No regularization
    print('\n--- No Regularization ---')
    model = MLP(layer_sizes, dropout=0.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    results['No Reg'], _ = train_with_optimizer(model, train_loader, test_loader, optimizer, epochs=20)
    
    # 2. L2 regularization
    print('\n--- L2 Regularization (weight_decay=0.001) ---')
    model = MLP(layer_sizes, dropout=0.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    results['L2 Reg'], _ = train_with_optimizer(model, train_loader, test_loader, optimizer, epochs=20)
    
    # 3. Dropout
    print('\n--- Dropout (p=0.5) ---')
    model = MLP(layer_sizes, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    results['Dropout'], _ = train_with_optimizer(model, train_loader, test_loader, optimizer, epochs=20)
    
    # need train accuracy too - quick re-run just to get it
    def get_train_acc(model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        return 100. * correct / total
    
    # plot
    plt.figure(figsize=(10, 5))
    markers = ['o-', 's-', '^-']
    
    for idx, (name, hist) in enumerate(results.items()):
        plt.plot(hist['test_acc'], markers[idx], label=name, linewidth=2, markersize=4)
    
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Regularization Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/part2c_regularization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # table
    print('\n=== Regularization Results ===')
    print(f'{"Method":<15} {"Final Test Acc":<15}')
    print('-'*30)
    for name in results:
        print(f'{name:<15} {results[name]["test_acc"][-1]:.2f}%')


def main():
    import os
    os.makedirs('results', exist_ok=True)
    
    part_a_optimizers()
    part_b_gradients()
    part_c_regularization()
    
    print('\n✓ All Part 2 results saved to results/ folder')


if __name__ == '__main__':
    main()


