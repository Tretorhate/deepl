"""
Training module for Generative Models (VAE and GAN)
Handles training and evaluation of both VAE and GAN architectures
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np


class VAETrainer:
    """Trainer for Variational Autoencoder (VAE)."""

    def __init__(self, model, config, device):
        """
        Args:
            model: VAE model
            config: Configuration dictionary with:
                - learning_rate
                - weight_decay (optional)
                - epochs
                - patience (early stopping)
            device: torch.device (cuda or cpu)
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4),
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('scheduler_factor', 0.5),
            patience=config.get('scheduler_patience', 5),
        )

        # Training history
        self.history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'val_loss': [],
            'val_recon_loss': [],
            'val_kl_loss': [],
        }

    def train_epoch(self, train_loader):
        """
        Train for one epoch.

        Args:
            train_loader: Training DataLoader

        Returns:
            Tuple of (avg_loss, avg_recon_loss, avg_kl_loss)
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for batch in tqdm(train_loader, desc='Training', leave=False):
            # Get images (handle dict or tensor)
            if isinstance(batch, dict):
                images = batch['image'].to(self.device)
            else:
                images = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            x_recon, mean, logvar, z = self.model(images)
            loss, recon_loss, kl_loss = self.model.vae_loss(x_recon, images, mean, logvar)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)

        return avg_loss, avg_recon_loss, avg_kl_loss

    def validate(self, val_loader):
        """
        Validate model.

        Args:
            val_loader: Validation DataLoader

        Returns:
            Tuple of (avg_loss, avg_recon_loss, avg_kl_loss)
        """
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating', leave=False):
                # Get images
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                else:
                    images = batch.to(self.device)

                # Forward pass
                x_recon, mean, logvar, z = self.model(images)
                loss, recon_loss, kl_loss = self.model.vae_loss(x_recon, images, mean, logvar)

                # Track metrics
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()

        avg_loss = total_loss / len(val_loader)
        avg_recon_loss = total_recon_loss / len(val_loader)
        avg_kl_loss = total_kl_loss / len(val_loader)

        return avg_loss, avg_recon_loss, avg_kl_loss

    def train(self, train_loader, val_loader):
        """
        Train VAE with early stopping.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader

        Returns:
            Training history dictionary
        """
        epochs = self.config.get('epochs', 50)
        patience = self.config.get('patience', 15)

        print(f"\nTraining VAE for {epochs} epochs with patience={patience}")

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_recon, train_kl = self.train_epoch(train_loader)

            # Validate
            val_loss, val_recon, val_kl = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_recon_loss'].append(train_recon)
            self.history['train_kl_loss'].append(train_kl)
            self.history['val_loss'].append(val_loss)
            self.history['val_recon_loss'].append(val_recon)
            self.history['val_kl_loss'].append(val_kl)

            # Print progress
            if epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch:3d}/{epochs} | "
                      f"Train: Loss={train_loss:.4f}, Recon={train_recon:.4f}, KL={train_kl:.4f} | "
                      f"Val: Loss={val_loss:.4f}, Recon={val_recon:.4f}, KL={val_kl:.4f} | "
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

    def generate_samples(self, num_samples=16):
        """
        Generate samples from the VAE.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Generated samples tensor
        """
        samples = self.model.generate(num_samples=num_samples, device=self.device)
        return samples

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


class GANTrainer:
    """Trainer for Generative Adversarial Network (GAN)."""

    def __init__(self, model, config, device):
        """
        Args:
            model: GAN model
            config: Configuration dictionary with:
                - learning_rate (or generator_lr and discriminator_lr)
                - epochs
            device: torch.device (cuda or cpu)
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.noise_dim = model.noise_dim

        # Optimizers for generator and discriminator
        gen_lr = config.get('generator_lr', config.get('learning_rate', 2e-4))
        dis_lr = config.get('discriminator_lr', config.get('learning_rate', 2e-4))
        beta1 = config.get('beta1', 0.5)

        self.optimizer_g = optim.Adam(
            model.generator.parameters(),
            lr=gen_lr,
            betas=(beta1, 0.999),
        )
        self.optimizer_d = optim.Adam(
            model.discriminator.parameters(),
            lr=dis_lr,
            betas=(beta1, 0.999),
        )

        # Loss function
        self.criterion = nn.BCELoss()

        # Training history
        self.history = {
            'gen_loss': [],
            'dis_loss': [],
            'dis_loss_real': [],
            'dis_loss_fake': [],
        }

    def train_epoch(self, train_loader):
        """
        Train for one epoch.

        Args:
            train_loader: Training DataLoader

        Returns:
            Tuple of (avg_gen_loss, avg_dis_loss)
        """
        self.model.train()
        total_gen_loss = 0
        total_dis_loss = 0
        total_dis_loss_real = 0
        total_dis_loss_fake = 0

        for batch in tqdm(train_loader, desc='Training GAN', leave=False):
            batch_size = batch['image'].size(0) if isinstance(batch, dict) else batch.size(0)

            # Get real images
            if isinstance(batch, dict):
                real_images = batch['image'].to(self.device)
            else:
                real_images = batch.to(self.device)

            # Labels for real and fake
            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)

            # ======== Train Discriminator ========
            self.optimizer_d.zero_grad()

            # Real images
            d_real_output = self.model.discriminate(real_images)
            d_loss_real = self.criterion(d_real_output, real_labels)

            # Fake images
            z = torch.randn(batch_size, self.noise_dim, device=self.device)
            fake_images = self.model.generator(z)
            d_fake_output = self.model.discriminate(fake_images.detach())
            d_loss_fake = self.criterion(d_fake_output, fake_labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.optimizer_d.step()

            # ======== Train Generator ========
            self.optimizer_g.zero_grad()

            # Generate fake images
            z = torch.randn(batch_size, self.noise_dim, device=self.device)
            fake_images = self.model.generator(z)
            d_fake_output = self.model.discriminate(fake_images)

            # Generator tries to fool discriminator
            g_loss = self.criterion(d_fake_output, real_labels)
            g_loss.backward()
            self.optimizer_g.step()

            # Track metrics
            total_gen_loss += g_loss.item()
            total_dis_loss += d_loss.item()
            total_dis_loss_real += d_loss_real.item()
            total_dis_loss_fake += d_loss_fake.item()

        avg_gen_loss = total_gen_loss / len(train_loader)
        avg_dis_loss = total_dis_loss / len(train_loader)
        avg_dis_loss_real = total_dis_loss_real / len(train_loader)
        avg_dis_loss_fake = total_dis_loss_fake / len(train_loader)

        return avg_gen_loss, avg_dis_loss, avg_dis_loss_real, avg_dis_loss_fake

    def train(self, train_loader):
        """
        Train GAN.

        Args:
            train_loader: Training DataLoader

        Returns:
            Training history dictionary
        """
        epochs = self.config.get('epochs', 50)

        print(f"\nTraining GAN for {epochs} epochs")

        for epoch in range(1, epochs + 1):
            # Train
            gen_loss, dis_loss, dis_loss_real, dis_loss_fake = self.train_epoch(train_loader)

            # Update history
            self.history['gen_loss'].append(gen_loss)
            self.history['dis_loss'].append(dis_loss)
            self.history['dis_loss_real'].append(dis_loss_real)
            self.history['dis_loss_fake'].append(dis_loss_fake)

            # Print progress
            if epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs:
                print(f"  Epoch {epoch:3d}/{epochs} | "
                      f"Gen Loss: {gen_loss:.4f} | "
                      f"Dis Loss: {dis_loss:.4f} (Real: {dis_loss_real:.4f}, Fake: {dis_loss_fake:.4f})")

        return self.history

    def generate_samples(self, num_samples=16):
        """
        Generate samples from the GAN.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Generated samples tensor
        """
        samples = self.model.generate(num_samples=num_samples, device=self.device)
        return samples

    def get_history(self):
        """Return training history."""
        return self.history

    def save_model(self, path_g, path_d):
        """Save models to paths."""
        torch.save(self.model.generator.state_dict(), path_g)
        torch.save(self.model.discriminator.state_dict(), path_d)
        print(f"Models saved to {path_g} and {path_d}")

    def load_model(self, path_g, path_d):
        """Load models from paths."""
        self.model.generator.load_state_dict(torch.load(path_g, map_location=self.device))
        self.model.discriminator.load_state_dict(torch.load(path_d, map_location=self.device))
        self.model.to(self.device)
        print(f"Models loaded from {path_g} and {path_d}")


if __name__ == '__main__':
    print("Generative trainers module imported successfully")
