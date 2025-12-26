"""
Problem 3: Convolutional Neural Networks
Parts A, B, C - CNN on CIFAR-10
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {device}')

CIFAR_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_cifar10():
    """Load CIFAR-10"""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=100, num_workers=0)
    
    return train_loader, test_loader


# ============== PART A: Basic CNN ==============

class BasicCNN(nn.Module):
    """CNN architecture from assignment"""
    def __init__(self):
        super().__init__()
        # conv layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32x32 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 16x16 -> 16x16
        
        self.pool = nn.MaxPool2d(2, 2)  # halves spatial dims
        self.relu = nn.ReLU()
        
        # after conv+pool: 64 * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 32->16
        x = self.pool(self.relu(self.conv2(x)))  # 16->8
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_cnn(model, train_loader, test_loader, epochs=20, lr=0.001):
    """Train CNN and return history"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'test_acc': []}
    start_time = time.time()
    
    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # eval
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        val_loss /= len(test_loader)
        test_acc = 100. * correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Test Acc: {test_acc:.2f}%')
    
    elapsed = time.time() - start_time
    return history, elapsed


def get_predictions(model, loader):
    """Get all predictions and labels"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            preds = model(data).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(preds, labels, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/part3a_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============== PART B: Architecture Experiments ==============

class CNN_V1(nn.Module):
    """Smaller - fewer filters"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


class CNN_V2(nn.Module):
    """Deeper - 3 conv layers"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


class CNN_V3(nn.Module):
    """Different pooling - avg pool, bigger kernels"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.AvgPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ============== PART C: Visualizations ==============

def visualize_filters(model):
    """Show first conv layer filters"""
    # get first conv layer weights
    weights = model.conv1.weight.data.cpu()
    
    # normalize for display
    weights = weights - weights.min()
    weights = weights / weights.max()
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:
            # weights shape: [out_channels, in_channels, H, W]
            # take first 8 filters, show as RGB
            filt = weights[i].permute(1, 2, 0).numpy()  # H, W, C
            ax.imshow(filt)
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')
    
    plt.suptitle('First Conv Layer Filters (8 of 32)')
    plt.tight_layout()
    plt.savefig('results/part3c_filters.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_activations(model, loader):
    """Show activation maps for sample images"""
    model.eval()
    
    # grab a few images
    data, labels = next(iter(loader))
    data = data[:3].to(device)
    
    # hook to grab activations
    activations = []
    def hook(module, input, output):
        activations.append(output.detach().cpu())
    
    handle = model.conv1.register_forward_hook(hook)
    _ = model(data)
    handle.remove()
    
    acts = activations[0]  # [batch, channels, H, W]
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    
    # unnormalize images for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2470, 0.2435, 0.2616])
    
    for row in range(3):
        # original image
        img = data[row].cpu()
        img = img * std.view(3, 1, 1) + mean.view(3, 1, 1)
        img = img.permute(1, 2, 0).numpy().clip(0, 1)
        
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f'Original: {CIFAR_CLASSES[labels[row]]}')
        axes[row, 0].axis('off')
        
        # 4 activation maps
        for i in range(4):
            act_map = acts[row, i].numpy()
            axes[row, i+1].imshow(act_map, cmap='viridis')
            axes[row, i+1].set_title(f'Filter {i+1}')
            axes[row, i+1].axis('off')
    
    plt.suptitle('Activation Maps from First Conv Layer')
    plt.tight_layout()
    plt.savefig('results/part3c_activations.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    import os
    os.makedirs('results', exist_ok=True)
    
    train_loader, test_loader = load_cifar10()
    
    # ===== Part A: Basic CNN =====
    print('\n' + '='*50)
    print('PART A: Basic CNN')
    print('='*50)
    
    model = BasicCNN()
    print(f'\nArchitecture:\n{model}')
    print(f'\nTotal params: {count_params(model):,}')
    
    history, train_time = train_cnn(model, train_loader, test_loader, epochs=20)
    
    # loss curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', linewidth=2)
    plt.plot(history['val_loss'], label='Validation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['test_acc'], linewidth=2, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/part3a_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f'\n>>> Final Test Accuracy: {history["test_acc"][-1]:.2f}%')
    
    # confusion matrix
    preds, labels = get_predictions(model, test_loader)
    plot_confusion_matrix(preds, labels, CIFAR_CLASSES)
    
    # ===== Part B: Architecture Experiments =====
    print('\n' + '='*50)
    print('PART B: Architecture Experimentation')
    print('='*50)
    
    architectures = {
        'V1 (Smaller)': CNN_V1(),
        'V2 (Deeper)': CNN_V2(),
        'V3 (AvgPool)': CNN_V3(),
        'Basic': BasicCNN()
    }
    
    arch_results = {}
    
    for name, arch_model in architectures.items():
        print(f'\n--- {name} ---')
        print(f'Params: {count_params(arch_model):,}')
        hist, t = train_cnn(arch_model, train_loader, test_loader, epochs=20)
        arch_results[name] = {
            'params': count_params(arch_model),
            'time': t,
            'acc': hist['test_acc'][-1],
            'history': hist
        }
    
    # comparison table
    print('\n' + '='*60)
    print(f'{"Architecture":<15} {"Params":<12} {"Time (s)":<12} {"Test Acc":<10}')
    print('-'*49)
    for name, res in arch_results.items():
        print(f'{name:<15} {res["params"]:<12,} {res["time"]:<12.1f} {res["acc"]:.2f}%')
    print('='*60)
    
    # plot comparison
    plt.figure(figsize=(10, 5))
    for name, res in arch_results.items():
        plt.plot(res['history']['test_acc'], label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Architecture Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/part3b_architecture_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ===== Part C: Visualizations =====
    print('\n' + '='*50)
    print('PART C: Visualizing Learned Features')
    print('='*50)
    
    # use the trained basic model
    visualize_filters(model)
    visualize_activations(model, test_loader)
    
    print('\n✓ All Part 3 results saved to results/ folder')


if __name__ == '__main__':
    main()


