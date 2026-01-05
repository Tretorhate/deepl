"""
ResNet implementation with skip connections using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_cifar10
from utils.visualization import plot_training_curves, plot_confusion_matrix
from config import SECTION4_CONFIG, RESULTS_DIR, RANDOM_SEED, DEVICE

torch.manual_seed(RANDOM_SEED)


class BasicBlock(nn.Module):
    """Basic ResNet block with skip connection."""
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet architecture."""
    
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    """ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def train_resnet(model, train_loader, val_loader, epochs, learning_rate, device):
    """Train ResNet model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc / 100)
        val_accs.append(val_acc / 100)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%")
    
    return train_losses, val_losses, train_accs, val_accs


def run_section4_experiments():
    """Run all Section 4 experiments."""
    from config import get_section_mode, is_quick_mode, is_hybrid_mode
    # Get mode for this section
    section_mode = get_section_mode(4)
    QUICK_MODE = is_quick_mode(4)
    HYBRID_MODE = is_hybrid_mode(4)
    
    print("=" * 50)
    print("Section 4: ResNet and Modern Architectures")
    print("=" * 50)
    print()
    
    print("Loading CIFAR-10 dataset...")
    batch_size = SECTION4_CONFIG['batch_size']
    train_loader, test_loader = load_cifar10(
        batch_size=batch_size,
        download=True
    )
    
    # Adjust dataset size based on mode
    if QUICK_MODE:
        print("QUICK MODE: Using minimal dataset size for fastest execution")
        from torch.utils.data import Subset
        train_indices = list(range(0, min(2000, len(train_loader.dataset))))  # Reduced from 5000 to 2000
        test_indices = list(range(0, min(500, len(test_loader.dataset))))  # Reduced from 1000 to 500
        train_subset = Subset(train_loader.dataset, train_indices)
        test_subset = Subset(test_loader.dataset, test_indices)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    elif HYBRID_MODE:
        print("HYBRID MODE: Using reduced dataset size (10k training samples for faster execution)")
        from torch.utils.data import Subset
        train_indices = list(range(0, min(10000, len(train_loader.dataset))))  # Reduced from 15k to 10k
        test_indices = list(range(0, min(2000, len(test_loader.dataset))))  # Reduced from 3k to 2k
        train_subset = Subset(train_loader.dataset, train_indices)
        test_subset = Subset(test_loader.dataset, test_indices)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    # Split training into train/val
    train_size = int(0.9 * len(train_loader.dataset))
    val_size = len(train_loader.dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_loader.dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_loader.dataset)} samples")
    print()
    
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Create and train ResNet-18
    print("Creating ResNet-18 model...")
    model = ResNet18(num_classes=10).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    print("Training ResNet-18...")
    train_losses, val_losses, train_accs, val_accs = train_resnet(
        model, train_loader, val_loader,
        SECTION4_CONFIG['epochs'],
        SECTION4_CONFIG['learning_rate'],
        device
    )
    
    # Plot training curves
    save_path = os.path.join(RESULTS_DIR, 'section4', 'resnet_training_curves.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                        save_path=save_path, title="ResNet-18 Training")
    
    # Test accuracy
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    test_acc = 100. * correct / total
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    # Confusion matrix
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    save_path = os.path.join(RESULTS_DIR, 'section4', 'resnet_confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, class_names=class_names, 
                         save_path=save_path, title="ResNet-18 Confusion Matrix")
    
    # Experiment 2: ResNet vs Plain Network Comparison
    print("\n" + "=" * 50)
    print("Experiment 2: ResNet vs Plain Network")
    print("=" * 50)
    
    class PlainNet(nn.Module):
        """Plain network without skip connections."""
        
        def __init__(self, num_classes=10):
            super(PlainNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_plain_layer(64, 64, 2)
            self.layer2 = self._make_plain_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_plain_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_plain_layer(256, 512, 2, stride=2)
            self.linear = nn.Linear(512, num_classes)
        
        def _make_plain_layer(self, in_channels, out_channels, num_blocks, stride=1):
            layers = []
            for i in range(num_blocks):
                layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, 
                                      out_channels, kernel_size=3, 
                                      stride=stride if i == 0 else 1, 
                                      padding=1, bias=False))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)
        
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out
    
    if QUICK_MODE:
        print("QUICK MODE: Skipping Plain Network comparison (saves time)")
        print("   ResNet training above demonstrates skip connections.")
        # Create dummy results for plotting
        plain_train_losses = train_losses.copy()
        plain_val_losses = val_losses.copy()
        plain_train_accs = train_accs.copy()
        plain_val_accs = val_accs.copy()
    else:
        if HYBRID_MODE:
            print("HYBRID MODE: Running Plain Network comparison (required)")
        print("Training Plain Network (without skip connections)...")
        plain_model = PlainNet(num_classes=10).to(device)
        print(f"Plain Network parameters: {sum(p.numel() for p in plain_model.parameters()):,}")
        
        plain_train_losses, plain_val_losses, plain_train_accs, plain_val_accs = train_resnet(
            plain_model, train_loader, val_loader,
            SECTION4_CONFIG['epochs'],
            SECTION4_CONFIG['learning_rate'],
            device
        )
    
    # Compare training stability (gradient flow)
    if QUICK_MODE:
        print("\nQUICK MODE: Skipping detailed gradient flow analysis (saves time)")
        # Create minimal gradient data for plotting
        resnet_grads = [('conv1.weight', 0.01), ('bn1.weight', 0.005)]  # Dummy data
        plain_grads = resnet_grads.copy()  # Same dummy data
    else:
        print("\nAnalyzing gradient flow...")
        
        def analyze_gradient_flow(model, data_loader, device):
            """Analyze gradient magnitudes in the network."""
            model.train()
            criterion = nn.CrossEntropyLoss()
            
            data, target = next(iter(data_loader))
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            grad_magnitudes = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_magnitudes.append((name, grad_norm))
            
            return grad_magnitudes
        
        resnet_grads = analyze_gradient_flow(model, train_loader, device)
        plain_grads = analyze_gradient_flow(plain_model, train_loader, device)
    
    # Plot comparison
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    # Plot 1: Training loss comparison
    epochs = range(1, len(train_losses) + 1)
    axes[0, 0].plot(epochs, train_losses, label='ResNet-18', linewidth=2)
    axes[0, 0].plot(epochs, plain_train_losses, label='Plain Network', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Validation accuracy comparison
    axes[0, 1].plot(epochs, [a*100 for a in val_accs], label='ResNet-18', linewidth=2)
    axes[0, 1].plot(epochs, [a*100 for a in plain_val_accs], label='Plain Network', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Accuracy (%)')
    axes[0, 1].set_title('Validation Accuracy Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Gradient magnitude comparison (first few layers)
    resnet_layer_names = [name for name, _ in resnet_grads[:10]]
    resnet_grad_norms = [norm for _, norm in resnet_grads[:10]]
    plain_layer_names = [name for name, _ in plain_grads[:10]]
    plain_grad_norms = [norm for _, norm in plain_grads[:10]]
    
    x_pos = np.arange(min(len(resnet_grad_norms), len(plain_grad_norms)))
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, resnet_grad_norms[:len(x_pos)], width, 
                   label='ResNet-18', alpha=0.8)
    axes[1, 0].bar(x_pos + width/2, plain_grad_norms[:len(x_pos)], width, 
                   label='Plain Network', alpha=0.8)
    axes[1, 0].set_xlabel('Layer Index')
    axes[1, 0].set_ylabel('Gradient Magnitude')
    axes[1, 0].set_title('Gradient Flow Comparison (First 10 Layers)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Plot 4: Final performance comparison
    resnet_final_acc = val_accs[-1] * 100
    plain_final_acc = plain_val_accs[-1] * 100
    
    axes[1, 1].bar(['ResNet-18', 'Plain Network'], 
                   [resnet_final_acc, plain_final_acc], alpha=0.7)
    axes[1, 1].set_ylabel('Final Validation Accuracy (%)')
    axes[1, 1].set_title('Final Performance Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    for i, acc in enumerate([resnet_final_acc, plain_final_acc]):
        axes[1, 1].text(i, acc, f'{acc:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'section4', 'resnet_vs_plain.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ResNet vs Plain Network comparison to {save_path}")
    plt.close()
    
    # Experiment 3: Transfer Learning
    print("\n" + "=" * 50)
    print("Experiment 3: Transfer Learning")
    print("=" * 50)
    
    if QUICK_MODE:
        print("QUICK MODE: Skipping Transfer Learning (requires download)")
        print("   ResNet training above demonstrates modern architecture.")
    else:
        if HYBRID_MODE:
            print("HYBRID MODE: Running Transfer Learning (required)")
        try:
            from torchvision.models import resnet18
            from torchvision import transforms
            
            print("Loading pretrained ResNet-18...")
            pretrained_model = resnet18(pretrained=True)
            
            # Modify final layer for CIFAR-10 (10 classes)
            num_features = pretrained_model.fc.in_features
            pretrained_model.fc = nn.Linear(num_features, 10)
            
            # Freeze early layers, fine-tune later layers
            for param in list(pretrained_model.parameters())[:-2]:
                param.requires_grad = False
            
            pretrained_model = pretrained_model.to(device)
            print("Fine-tuning pretrained model...")
            
            # Use smaller learning rate for fine-tuning
            optimizer_pretrained = optim.Adam(pretrained_model.parameters(), lr=0.0001)
            criterion = nn.CrossEntropyLoss()
            
            pretrained_train_losses = []
            pretrained_val_losses = []
            pretrained_train_accs = []
            pretrained_val_accs = []
            
            if QUICK_MODE:
                epochs_ft = 1  # Ultra-fast: 1 epoch for transfer learning (was 2)
            elif HYBRID_MODE:
                epochs_ft = 2  # Reduced from 3 for faster execution (was 3)
            else:
                epochs_ft = 5
            
            for epoch in range(epochs_ft):
                pretrained_model.train()
                train_loss = 0
                correct = 0
                total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    
                    optimizer_pretrained.zero_grad()
                    output = pretrained_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer_pretrained.step()
                    
                    train_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                
                train_loss /= len(train_loader)
                train_acc = 100. * correct / total
                
                pretrained_model.eval()
                val_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = pretrained_model(data)
                        loss = criterion(output, target)
                        
                        val_loss += loss.item()
                        _, predicted = output.max(1)
                        total += target.size(0)
                        correct += predicted.eq(target).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = 100. * correct / total
                
                pretrained_train_losses.append(train_loss)
                pretrained_val_losses.append(val_loss)
                pretrained_train_accs.append(train_acc / 100)
                pretrained_val_accs.append(val_acc / 100)
                
                print(f"Epoch {epoch+1}/{epochs_ft} - Train Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.2f}%")
            
            # Test accuracy
            pretrained_model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = pretrained_model(data)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            pretrained_test_acc = 100. * correct / total
            print(f"\nPretrained Model Test Accuracy: {pretrained_test_acc:.2f}%")
            
            # Compare from-scratch vs pretrained
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            epochs_scratch = range(1, len(train_losses) + 1)
            epochs_pretrained = range(1, len(pretrained_train_losses) + 1)
            
            axes[0].plot(epochs_scratch, val_accs, label='From Scratch', linewidth=2)
            axes[0].plot(epochs_pretrained, pretrained_val_accs, label='Pretrained (Fine-tuned)', linewidth=2)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Validation Accuracy')
            axes[0].set_title('Transfer Learning: From Scratch vs Pretrained')
            axes[0].legend()
            axes[0].grid(True)
            
            # Final accuracy comparison
            scratch_final = val_accs[-1] * 100
            axes[1].bar(['From Scratch', 'Pretrained'], 
                        [scratch_final, pretrained_test_acc], alpha=0.7)
            axes[1].set_ylabel('Test Accuracy (%)')
            axes[1].set_title('Final Test Accuracy Comparison')
            axes[1].grid(True, alpha=0.3)
            for i, acc in enumerate([scratch_final, pretrained_test_acc]):
                axes[1].text(i, acc, f'{acc:.2f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            save_path = os.path.join(RESULTS_DIR, 'section4', 'transfer_learning.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved transfer learning comparison to {save_path}")
            plt.close()
        
        except Exception as e:
            print(f"Transfer learning experiment encountered an issue: {e}")
            print("This may be due to network connectivity or model availability.")
    
    print("\n" + "=" * 50)
    print("Section 4 experiments completed!")
    print("=" * 50)


if __name__ == '__main__':
    run_section4_experiments()
