"""
Data loaders for MNIST, Fashion-MNIST, and CIFAR-10 datasets.
"""

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DATA_DIR


def load_mnist(batch_size=64, download=True):
    """
    Load MNIST dataset.
    
    Args:
        batch_size: Batch size for data loaders
        download: Whether to download the dataset if not present
    
    Returns:
        train_loader, test_loader: PyTorch DataLoaders
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=download,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=download,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def load_fashion_mnist(batch_size=64, download=True):
    """
    Load Fashion-MNIST dataset.
    
    Args:
        batch_size: Batch size for data loaders
        download: Whether to download the dataset if not present
    
    Returns:
        train_loader, test_loader: PyTorch DataLoaders
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=download,
        transform=transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=download,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def load_cifar10(batch_size=32, download=True):
    """
    Load CIFAR-10 dataset.
    
    Args:
        batch_size: Batch size for data loaders
        download: Whether to download the dataset if not present
    
    Returns:
        train_loader, test_loader: PyTorch DataLoaders
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=download,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=download,
        transform=transform_test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def load_mnist_numpy():
    """
    Load MNIST as NumPy arrays (for NumPy-based implementations).
    
    Returns:
        X_train, y_train, X_test, y_test: NumPy arrays
    """
    # Use a large batch size to get all data efficiently
    train_loader, test_loader = load_mnist(batch_size=60000, download=True)
    
    # Get all training data (iterate through all batches to ensure we get all 60k samples)
    X_train_list = []
    y_train_list = []
    for X_batch, y_batch in train_loader:
        X_train_list.append(X_batch.numpy())
        y_train_list.append(y_batch.numpy())
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    # Get all test data (iterate through all batches to ensure we get all 10k samples)
    X_test_list = []
    y_test_list = []
    for X_batch, y_batch in test_loader:
        X_test_list.append(X_batch.numpy())
        y_test_list.append(y_batch.numpy())
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    # Reshape to (N, 784) for MLP
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    return X_train, y_train, X_test, y_test
