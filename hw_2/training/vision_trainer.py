"""
Vision Model Trainer
Handles training and evaluation of image classification models
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class VisionTrainer:
    """Trainer for vision/image classification models with early stopping and LR scheduling."""

    def __init__(self, model, config, device):
        """
        Args:
            model: PyTorch model to train
            config: Configuration dictionary with:
                - learning_rate
                - weight_decay (L2 regularization)
                - epochs
                - patience (early stopping)
                - Optional: scheduler_factor, scheduler_patience
            device: torch.device (cuda or cpu)
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer with weight decay (L2 regularization)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4),
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('scheduler_factor', 0.5),
            patience=config.get('scheduler_patience', 5),
        )

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

    def train_epoch(self, train_loader):
        """
        Train for one epoch.

        Args:
            train_loader: Training DataLoader

        Returns:
            Tuple of (avg_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc='Training', leave=False):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader):
        """
        Validate model.

        Args:
            val_loader: Validation DataLoader

        Returns:
            Tuple of (avg_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating', leave=False):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def train(self, train_loader, val_loader):
        """
        Train model with early stopping.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader

        Returns:
            Training history dictionary
        """
        epochs = self.config['epochs']
        patience = self.config.get('patience', 15)

        print(f"\nTraining for {epochs} epochs with patience={patience}")

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Print progress
            if epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch:3d}/{epochs} | "
                      f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
                      f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}% | "
                      f"LR={lr:.2e}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"  Restored best model (val_loss={self.best_val_loss:.4f})")

        return self.history

    def evaluate(self, test_loader, class_names=None):
        """
        Evaluate model on test set.

        Args:
            test_loader: Test DataLoader
            class_names: Optional list of class names

        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating', leave=False):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Calculate metrics
        accuracy = 100 * (all_predictions == all_labels).sum() / len(all_labels)
        loss = total_loss / len(test_loader)

        # Per-class metrics
        results = {
            'accuracy': accuracy,
            'loss': loss,
            'predictions': all_predictions,
            'labels': all_labels,
        }

        # Per-class accuracy
        if class_names is not None:
            for i, class_name in enumerate(class_names):
                class_mask = all_labels == i
                if class_mask.sum() > 0:
                    class_acc = 100 * (all_predictions[class_mask] == all_labels[class_mask]).sum() / class_mask.sum()
                    results[f'{class_name}_acc'] = class_acc

        return results

    def get_history(self):
        """Return training history."""
        return self.history

    def save_model(self, path):
        """Save model to path."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model from path."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")


if __name__ == '__main__':
    print("Vision trainer module imported successfully")
