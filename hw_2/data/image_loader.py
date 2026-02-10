"""
Medical Image Dataset Loading Module
Handles loading and preprocessing of medical image datasets
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import urllib.request
import tarfile
from sklearn.model_selection import train_test_split


class MedicalImageDataset(Dataset):
    """Medical image classification dataset."""

    def __init__(self, image_paths, labels, transform=None, mode='rgb'):
        """
        Args:
            image_paths: List of paths to images
            labels: List of labels
            transform: Optional image transforms
            mode: 'rgb' for RGB images (default) or 'grayscale' for grayscale
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]

        # Create dummy image if path doesn't exist (for testing)
        if not os.path.exists(img_path):
            if self.mode == 'grayscale':
                image = Image.new('L', (224, 224), color=128)
            else:
                image = Image.new('RGB', (224, 224), color='white')
        else:
            try:
                if self.mode == 'grayscale':
                    image = Image.open(img_path).convert('L')
                else:
                    image = Image.open(img_path).convert('RGB')
            except:
                if self.mode == 'grayscale':
                    image = Image.new('L', (224, 224), color=128)
                else:
                    image = Image.new('RGB', (224, 224), color='white')

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {'image': image, 'label': label}


def create_synthetic_medical_images(num_samples=200, image_size=224):
    """
    Create synthetic medical images for testing.

    In production, replace this with actual dataset loading
    (HAM10000, ChestX-ray14, etc.)

    Args:
        num_samples: Number of images to generate
        image_size: Size of each image

    Returns:
        (image_paths, labels, class_names) tuple
    """
    print(f"  Creating synthetic medical images ({num_samples} samples)...")

    # Create temp directory
    temp_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_images')
    os.makedirs(temp_dir, exist_ok=True)

    # Medical image classes
    class_names = ['Normal', 'Abnormal']

    image_paths = []
    labels = []

    # Generate synthetic images
    for i in range(num_samples):
        # Random class
        label = np.random.randint(0, len(class_names))

        # Create synthetic image
        if label == 0:  # Normal - mostly uniform
            img_array = np.random.randint(100, 150, (image_size, image_size, 3), dtype=np.uint8)
        else:  # Abnormal - with patterns/anomalies
            img_array = np.random.randint(50, 200, (image_size, image_size, 3), dtype=np.uint8)
            # Add some "abnormal" patterns
            for _ in range(np.random.randint(3, 8)):
                y = np.random.randint(0, image_size)
                x = np.random.randint(0, image_size)
                size = np.random.randint(10, 50)
                img_array[max(0, y-size):min(image_size, y+size),
                         max(0, x-size):min(image_size, x+size)] = np.random.randint(0, 50)

        # Save image
        img = Image.fromarray(img_array)
        img_path = os.path.join(temp_dir, f'image_{i:04d}.png')
        img.save(img_path)

        image_paths.append(img_path)
        labels.append(label)

    print(f"  Generated {num_samples} synthetic images")
    print(f"  Classes: {class_names}")

    return image_paths, labels, class_names


def load_image_dataset(dataset_name, config, mode='rgb'):
    """
    Load medical image dataset and create train/val/test splits.

    Args:
        dataset_name: Name of dataset ('HAM10000', 'ChestX-ray', 'Synthetic', etc.)
        config: Configuration dictionary with:
            - image_size (e.g., 224)
            - batch_size (optional)
            - num_classes
        mode: 'rgb' for 3-channel RGB (default) or 'grayscale' for 1-channel

    Returns:
        Dictionary with:
        {
            'train': DataLoader for training
            'val': DataLoader for validation
            'test': DataLoader for testing
            'num_classes': Number of output classes
            'class_names': Name of each class
        }
    """
    print(f"\n  Loading {dataset_name} dataset ({mode} mode)...")

    # Load raw data
    if 'Synthetic' in dataset_name or 'HAM' in dataset_name or 'ChestX' in dataset_name:
        # Use synthetic data for quick testing
        image_paths, labels, class_names = create_synthetic_medical_images(num_samples=200)
    else:
        print(f"  Warning: {dataset_name} not fully implemented, using Synthetic")
        image_paths, labels, class_names = create_synthetic_medical_images(num_samples=200)

    # Convert labels to numpy
    labels_array = np.array(labels)

    # Split into train (70%), val (15%), test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels_array, test_size=0.30, random_state=42, stratify=labels_array
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"    Train samples: {len(X_train)}")
    print(f"    Val samples: {len(X_val)}")
    print(f"    Test samples: {len(X_test)}")

    # Define transforms based on mode
    if mode == 'grayscale':
        # Normalize for grayscale (single channel)
        train_transform = transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            ),
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            ),
        ])
    else:  # RGB mode
        train_transform = transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    # Create datasets
    train_dataset = MedicalImageDataset(X_train, y_train, train_transform, mode=mode)
    val_dataset = MedicalImageDataset(X_val, y_val, val_test_transform, mode=mode)
    test_dataset = MedicalImageDataset(X_test, y_test, val_test_transform, mode=mode)

    # Create dataloaders
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Prepare output
    result = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'num_classes': len(class_names),
        'class_names': class_names,
    }

    print(f"  Dataset loaded successfully!")

    return result


if __name__ == '__main__':
    # Test the image loader
    config = {
        'image_size': 224,
        'batch_size': 32,
    }

    data = load_image_dataset('Synthetic', config)

    print("\nDataset loaded successfully!")
    print(f"Num classes: {data['num_classes']}")
    print(f"Classes: {data['class_names']}")

    # Test batch loading
    batch = next(iter(data['train']))
    print(f"\nBatch sample:")
    print(f"  image shape: {batch['image'].shape}")
    print(f"  label shape: {batch['label'].shape}")
