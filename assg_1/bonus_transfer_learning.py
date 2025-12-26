"""
BONUS: Transfer Learning (2 points)
Fine-tune pretrained ResNet18 on CIFAR-10
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {device}')


def load_cifar10_for_pretrained():
    """CIFAR-10 with transforms suitable for pretrained models"""
    # pretrained models expect 224x224 and ImageNet normalization
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    # smaller batch size cuz images are bigger now
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=32, num_workers=0)
    
    return train_loader, test_loader


def create_resnet_model():
    """Load pretrained ResNet18 and modify for CIFAR-10"""
    # load pretrained weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # freeze early layers (optional - can try both ways)
    for param in model.parameters():
        param.requires_grad = False
    
    # replace final fc layer for 10 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    
    return model


def train_transfer(model, train_loader, test_loader, epochs=10, lr=0.001):
    """Train transfer learning model"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # only train the final layer (unfrozen params)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    history = {'train_loss': [], 'test_acc': []}
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # progress
            if batch_idx % 200 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}', end='\r')
        
        train_loss /= len(train_loader)
        
        # eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        test_acc = 100. * correct / total
        history['train_loss'].append(train_loss)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%')
    
    elapsed = time.time() - start
    return history, elapsed


def main():
    import os
    os.makedirs('results', exist_ok=True)
    
    print('\n' + '='*50)
    print('BONUS: Transfer Learning with ResNet18')
    print('='*50)
    
    train_loader, test_loader = load_cifar10_for_pretrained()
    
    # create model
    model = create_resnet_model()
    
    # count trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal params: {total_params:,}')
    print(f'Trainable params: {trainable_params:,}')
    print(f'Frozen params: {total_params - trainable_params:,}')
    
    # train
    print('\nTraining (only final layer)...')
    history, elapsed = train_transfer(model, train_loader, test_loader, epochs=10)
    
    print(f'\n>>> Final Test Accuracy: {history["test_acc"][-1]:.2f}%')
    print(f'>>> Training Time: {elapsed:.1f}s')
    
    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Transfer Learning - Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['test_acc'], linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Transfer Learning - Test Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/bonus_transfer_learning.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # comparison with basic CNN (if you ran problem3 first)
    print('\n=== Comparison ===')
    print('Transfer Learning (ResNet18) should achieve ~80-85% accuracy')
    print('vs Basic CNN from Problem 3A which gets ~65-70%')
    print('\nWhy transfer learning helps:')
    print('- Pretrained features from ImageNet generalize well')
    print('- ResNet already knows edges, textures, shapes')
    print('- We only fine-tune the classifier on top')


if __name__ == '__main__':
    main()


