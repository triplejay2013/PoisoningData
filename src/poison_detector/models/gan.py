"""
GAN Implementation for Poison Data Detection.

This module implements a simple Generative Adversarial Network (GAN) for generating
synthetic images and detecting poisoned data anomalies. GANs consist of two neural networks
that compete against each other: a generator and a discriminator.

- **Generator:** Produces synthetic images that mimic the real dataset.
- **Discriminator:** Attempts to distinguish between real and synthetic images.

### Poison Data Detection with GANs

The GAN is leveraged for poisoned data detection as follows:
1. **Synthetic Data Generation:** The generator learns to create images that resemble normal
   data in the training set.
2. **Discriminator's Role:** By learning to separate real from fake data, the discriminator
   identifies anomalies. Poisoned data, which deviates from the normal data distribution,
   tends to result in a higher discriminator loss.

The generator and discriminator are trained iteratively. Once trained, the discriminator
can be used as a tool to detect anomalies based on how confidently it classifies an image
as "real" or "fake".

This implementation is designed with flexibility to support CPU-based debugging by scaling
down model sizes and epochs.

### Components:
- `Generator`: The neural network responsible for generating synthetic images.
- `Discriminator`: The neural network responsible for classifying images as real or fake.
- `train_gan`: The training loop that alternates between training the generator and discriminator.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class Generator(nn.Module):
    """
    A simple generator network for producing synthetic images.

    Args:
        latent_dim (int): The size of the latent space (input noise vector).

    Attributes:
        model (nn.Sequential): The feedforward layers of the generator.
    """
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3 * 32 * 32),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Forward pass of the generator.

        Args:
            z (torch.Tensor): Latent space vector of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Generated images of shape (batch_size, 3, 32, 32).
        """
        return self.model(z).view(-1, 3, 32, 32)


class Discriminator(nn.Module):
    """
    A simple discriminator network for classifying images as real or fake.

    Attributes:
        model (nn.Sequential): The feedforward layers of the discriminator.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): Images of shape (batch_size, 3, 32, 32).

        Returns:
            torch.Tensor: Discriminator scores of shape (batch_size, 1).
        """
        return self.model(x)


def train_gan(train_loader, latent_dim=100, epochs=5, debug=False):
    """
    Train a simple GAN for generating synthetic data and anomaly detection.

    Args:
        train_loader (DataLoader): DataLoader containing the training dataset.
        latent_dim (int): Dimension of the latent space.
        epochs (int): Number of training epochs.
        debug (bool): If True, uses a smaller model and fewer epochs for debugging.

    Returns:
        Generator: The trained Generator model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gen = Generator(latent_dim).to(device)
    disc = Discriminator().to(device)

    optimizer_g = optim.Adam(gen.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(disc.parameters(), lr=0.0002)
    criterion = nn.BCELoss()

    for epoch in range(epochs if not debug else 1):
        for i, (real_images, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train Discriminator
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = gen(z)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            disc_loss = criterion(disc(real_images), real_labels) + criterion(disc(fake_images.detach()), fake_labels)

            optimizer_d.zero_grad()
            disc_loss.backward()
            optimizer_d.step()

            # Train Generator
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = gen(z)
            gen_loss = criterion(disc(fake_images), real_labels)

            optimizer_g.zero_grad()
            gen_loss.backward()
            optimizer_g.step()

        print(f"Epoch {epoch + 1}/{epochs}: Gen Loss: {gen_loss.item()}, Disc Loss: {disc_loss.item()}")

    return gen, disc
