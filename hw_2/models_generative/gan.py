"""
Generative Adversarial Network (GAN) for Medical Image Generation
Implements a standard GAN with convolutional generator and discriminator
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator network for GAN.

    Takes random noise as input and generates realistic images.
    Uses transposed convolutions to upsample from noise vector to image.
    """

    def __init__(self, noise_dim=100, output_channels=1, hidden_dim=256):
        """
        Args:
            noise_dim: Dimension of input noise vector
            output_channels: Number of output channels (1 for grayscale, 3 for RGB)
            hidden_dim: Number of hidden channels in conv layers
        """
        super(Generator, self).__init__()

        self.noise_dim = noise_dim

        # Project noise to spatial dimensions: noise_dim -> hidden_dim * 8 * 7 * 7
        self.fc = nn.Linear(noise_dim, hidden_dim * 8 * 7 * 7)

        # Transposed convolutions for upsampling
        self.model = nn.Sequential(
            # (batch, hidden_dim*8, 7, 7)
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            # (batch, hidden_dim*4, 14, 14)

            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            # (batch, hidden_dim*2, 28, 28)

            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            # (batch, hidden_dim, 56, 56)

            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            # (batch, hidden_dim//2, 112, 112)

            nn.ConvTranspose2d(hidden_dim // 2, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            # (batch, output_channels, 224, 224)
        )

    def forward(self, z):
        """
        Generate image from noise vector.

        Args:
            z: Noise vector (batch, noise_dim)

        Returns:
            x: Generated image (batch, output_channels, 224, 224)
        """
        h = self.fc(z)
        h = h.view(h.size(0), -1, 7, 7)
        x = self.model(h)
        return x


class Discriminator(nn.Module):
    """
    Discriminator network for GAN.

    Classifies images as real or fake using convolutional layers.
    """

    def __init__(self, input_channels=1, hidden_dim=256):
        """
        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            hidden_dim: Number of hidden channels in conv layers
        """
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # (batch, input_channels, 224, 224)
            nn.Conv2d(input_channels, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            # (batch, hidden_dim//2, 112, 112)

            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2),
            # (batch, hidden_dim, 56, 56)

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            # (batch, hidden_dim*2, 28, 28)

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            # (batch, hidden_dim*4, 14, 14)

            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2),
            # (batch, hidden_dim*8, 7, 7)
        )

        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 8 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Classify image as real or fake.

        Args:
            x: Image (batch, input_channels, 224, 224)

        Returns:
            logit: Probability of being real (batch, 1)
        """
        h = self.model(x)
        h = h.view(h.size(0), -1)
        logit = self.fc(h)
        return logit


class GAN(nn.Module):
    """
    Generative Adversarial Network combining Generator and Discriminator.
    """

    def __init__(self, noise_dim=100, output_channels=1, hidden_dim=256):
        """
        Args:
            noise_dim: Dimension of input noise vector
            output_channels: Number of output channels
            hidden_dim: Number of hidden channels
        """
        super(GAN, self).__init__()

        self.noise_dim = noise_dim
        self.generator = Generator(noise_dim, output_channels, hidden_dim)
        self.discriminator = Discriminator(output_channels, hidden_dim)

    def forward(self, z):
        """
        Generate images from noise.

        Args:
            z: Noise vector (batch, noise_dim)

        Returns:
            generated: Generated images
        """
        generated = self.generator(z)
        return generated

    def discriminate(self, x):
        """
        Classify images as real or fake.

        Args:
            x: Image (batch, channels, 224, 224)

        Returns:
            logit: Probability of being real
        """
        logit = self.discriminator(x)
        return logit

    def generate(self, num_samples=16, device='cpu'):
        """
        Generate new samples.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate on

        Returns:
            generated: Generated images
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.noise_dim, device=device)
            generated = self.generator(z)

        return generated


if __name__ == '__main__':
    print("Testing GAN...")

    # Create model
    gan = GAN(noise_dim=100, output_channels=1, hidden_dim=256)

    # Test generator
    z = torch.randn(4, 100)
    generated = gan.generator(z)
    print(f"Noise shape: {z.shape}")
    print(f"Generated image shape: {generated.shape}")

    # Test discriminator
    real = torch.randn(4, 1, 224, 224)
    d_output_real = gan.discriminator(real)
    d_output_fake = gan.discriminator(generated.detach())
    print(f"\nDiscriminator output (real): {d_output_real.shape}")
    print(f"Discriminator output (fake): {d_output_fake.shape}")
    print(f"Real scores (should be ~1): {d_output_real.mean():.4f}")
    print(f"Fake scores (should be ~0): {d_output_fake.mean():.4f}")

    # Test generation
    samples = gan.generate(num_samples=8)
    print(f"\nGenerated samples shape: {samples.shape}")

    print("\nGAN test passed!")
