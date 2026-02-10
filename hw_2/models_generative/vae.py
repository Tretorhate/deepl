"""
Variational Autoencoder (VAE) for Medical Image Generation
Implements a VAE with encoder-decoder architecture for unsupervised learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Variational Autoencoder for generating medical images.

    Architecture:
    - Encoder: Convolutional layers → latent distribution (mean, logvar)
    - Latent space: Reparameterization trick for sampling
    - Decoder: Transposed convolutions → reconstructed image
    """

    def __init__(self, input_channels=1, latent_dim=32, hidden_dim=128,
                 reconstruction_loss='mse', kl_weight=1.0):
        """
        Args:
            input_channels: Number of input image channels (1 for grayscale, 3 for RGB)
            latent_dim: Dimension of latent space
            hidden_dim: Number of hidden channels in conv layers
            reconstruction_loss: 'mse' or 'bce' loss
            kl_weight: Weight for KL divergence in loss
        """
        super(VAE, self).__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.reconstruction_loss_type = reconstruction_loss
        self.kl_weight = kl_weight

        # ============ ENCODER ============
        # Expects input: (batch, channels, 224, 224)
        self.encoder = nn.Sequential(
            # (batch, input_channels, 224, 224)
            nn.Conv2d(input_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            # (batch, hidden_dim, 112, 112)

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            # (batch, hidden_dim*2, 56, 56)

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            # (batch, hidden_dim*4, 28, 28)

            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(),
            # (batch, hidden_dim*8, 14, 14)
        )

        # Flatten: (batch, hidden_dim*8, 14, 14) -> (batch, hidden_dim*8*14*14)
        self.encoder_output_dim = hidden_dim * 8 * 14 * 14

        # Latent space projections
        self.fc_mean = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_dim, latent_dim)

        # ============ DECODER ============
        # From latent to spatial: latent_dim -> hidden_dim*8*14*14
        self.fc_decode = nn.Linear(latent_dim, self.encoder_output_dim)

        # Transposed convolutions for upsampling
        self.decoder = nn.Sequential(
            # (batch, hidden_dim*8, 14, 14)
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            # (batch, hidden_dim*4, 28, 28)

            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            # (batch, hidden_dim*2, 56, 56)

            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            # (batch, hidden_dim, 112, 112)

            nn.ConvTranspose2d(hidden_dim, input_channels, kernel_size=4, stride=2, padding=1),
            # (batch, input_channels, 224, 224)
        )

        # Output activation based on loss type
        if reconstruction_loss == 'bce':
            self.output_activation = nn.Sigmoid()
        else:  # mse
            self.output_activation = nn.Tanh()  # Or nn.Identity() for unbounded

    def encode(self, x):
        """Encode image to latent space."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten

        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)

        return mean, logvar

    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick: sample z ~ N(mean, exp(logvar))

        Args:
            mean: Mean of the latent distribution
            logvar: Log variance of the latent distribution

        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z):
        """Decode latent vector to image."""
        h = self.fc_decode(z)
        h = h.view(h.size(0), self.hidden_dim * 8, 14, 14)

        x_recon = self.decoder(h)
        x_recon = self.output_activation(x_recon)

        return x_recon

    def forward(self, x):
        """
        Forward pass: encode, sample, decode

        Args:
            x: Input image (batch, channels, 224, 224)

        Returns:
            x_recon: Reconstructed image
            mean: Mean of latent distribution
            logvar: Log variance of latent distribution
            z: Sampled latent vector
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)

        return x_recon, mean, logvar, z

    def vae_loss(self, x_recon, x_original, mean, logvar):
        """
        VAE loss = Reconstruction loss + KL divergence

        Args:
            x_recon: Reconstructed image
            x_original: Original image
            mean: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            loss: Total VAE loss
            recon_loss: Reconstruction loss component
            kl_loss: KL divergence component
        """
        # Reconstruction loss
        if self.reconstruction_loss_type == 'bce':
            recon_loss = F.binary_cross_entropy(x_recon, x_original, reduction='mean')
        else:  # mse
            recon_loss = F.mse_loss(x_recon, x_original, reduction='mean')

        # KL divergence loss: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        # This is the KL divergence between N(mean, exp(logvar)) and N(0, 1)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        # Total loss
        loss = recon_loss + self.kl_weight * kl_loss

        return loss, recon_loss, kl_loss

    def generate(self, num_samples=16, device='cpu'):
        """
        Generate new samples from the latent space.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate on (cpu or cuda)

        Returns:
            generated: Generated images (num_samples, channels, 224, 224)
        """
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.latent_dim, device=device)
            generated = self.decode(z)

        return generated


if __name__ == '__main__':
    print("Testing VAE...")

    # Create model
    vae = VAE(input_channels=1, latent_dim=32, hidden_dim=128)

    # Test forward pass
    x = torch.randn(4, 1, 224, 224)
    x_recon, mean, logvar, z = vae(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {x_recon.shape}")
    print(f"Latent mean shape: {mean.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    print(f"Sampled z shape: {z.shape}")

    # Test loss
    loss, recon_loss, kl_loss = vae.vae_loss(x_recon, x, mean, logvar)
    print(f"\nLoss breakdown:")
    print(f"  Total loss: {loss:.4f}")
    print(f"  Reconstruction loss: {recon_loss:.4f}")
    print(f"  KL loss: {kl_loss:.4f}")

    # Test generation
    generated = vae.generate(num_samples=8)
    print(f"\nGenerated samples shape: {generated.shape}")

    print("\nVAE test passed!")
