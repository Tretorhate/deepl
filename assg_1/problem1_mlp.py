"""
Problem 1: Implementing and Training MLPs
Parts A, B, C - MLP on MNIST + XOR
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {device}')


# ============== PART A: MLP Implementation ==============

class MLP(nn.Module):
    """Flexible MLP - works with any number of layers"""
    
    def __init__(self, layer_sizes, activation='relu'):
        super(MLP, self).__init__()
        
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        self.activation = activations[activation]
        
        # build layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # no activation after last layer
                layers.append(self.activation)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.network(x)


def load_mnist():
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)
    
    return train_loader, test_loader


def train(model, train_loader, test_loader, epochs=20, lr=0.01):
    """Train and return history"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        test_acc = evaluate(model, test_loader)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%')
    
    return history


def evaluate(model, loader):
    """Get accuracy"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return 100. * correct / total


# ============== PART C: XOR ==============

class XOR_Net(nn.Module):
    """Simple network for XOR"""
    def __init__(self, hidden=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


def train_xor():
    """Train on XOR problem"""
    X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    
    model = XOR_Net(hidden=4)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    losses = []
    for epoch in range(1000):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return model, X, y, losses


def plot_xor_boundary(model, X, y):
    """Plot decision boundary for XOR"""
    h = 0.01
    xx, yy = np.meshgrid(np.arange(-0.5, 1.5, h), np.arange(-0.5, 1.5, h))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    
    with torch.no_grad():
        Z = model(grid).numpy().reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.8)
    plt.colorbar(label='Output')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    colors = ['red' if yi == 0 else 'blue' for yi in y.numpy().flatten()]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidth=2, zorder=5)
    
    for i in range(len(X)):
        plt.annotate(f'{int(y[i][0])}', (X[i][0], X[i][1]), 
                    textcoords='offset points', xytext=(0, 12), ha='center', fontsize=12, fontweight='bold')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('XOR Decision Boundary')
    plt.savefig('results/part1c_xor_boundary.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    import os
    os.makedirs('results', exist_ok=True)
    
    train_loader, test_loader = load_mnist()
    layer_sizes = [784, 128, 64, 10]  # required architecture
    
    # ===== Part A: Train MLP with ReLU =====
    print('\n' + '='*50)
    print('PART A: MLP Implementation')
    print('='*50)
    
    model = MLP(layer_sizes, activation='relu')
    print(f'\nModel architecture: {layer_sizes}')
    print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')
    
    history_relu = train(model, train_loader, test_loader, epochs=20)
    
    print(f'\n>>> Final Train Acc: {history_relu["train_acc"][-1]:.2f}%')
    print(f'>>> Final Test Acc: {history_relu["test_acc"][-1]:.2f}%')
    
    # ===== Part B: Compare activations =====
    print('\n' + '='*50)
    print('PART B: Activation Function Comparison')
    print('='*50)
    
    results = {}
    for act in ['relu', 'sigmoid', 'tanh']:
        print(f'\n--- Training with {act.upper()} ---')
        model = MLP(layer_sizes, activation=act)
        results[act] = train(model, train_loader, test_loader, epochs=20)
    
    # plot loss curves
    plt.figure(figsize=(10, 5))
    for act in results:
        plt.plot(results[act]['train_loss'], label=act.upper(), linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss - Different Activations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/part1b_loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # comparison table
    print('\n' + '='*40)
    print(f'{"Activation":<12} {"Test Accuracy":<15}')
    print('-'*27)
    for act in results:
        print(f'{act.upper():<12} {results[act]["test_acc"][-1]:.2f}%')
    print('='*40)
    
    # ===== Part C: XOR =====
    print('\n' + '='*50)
    print('PART C: XOR Problem')
    print('='*50)
    
    xor_model, X, y, losses = train_xor()
    
    # test accuracy
    with torch.no_grad():
        preds = (xor_model(X) > 0.5).float()
        acc = (preds == y).float().mean().item() * 100
    
    print(f'\nXOR Results:')
    print(f'{"Input":<10} {"Expected":<10} {"Predicted"}')
    print('-'*30)
    for i in range(4):
        print(f'({int(X[i][0])}, {int(X[i][1])})     {int(y[i][0]):<10} {int(preds[i][0])}')
    print(f'\n>>> Accuracy: {acc:.0f}%')
    
    plot_xor_boundary(xor_model, X, y)
    
    # loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('XOR Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/part1c_xor_loss.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print('\n✓ All Part 1 results saved to results/ folder')


if __name__ == '__main__':
    main()


