"""
GAN Anomaly Detection for Poisoned CIFAR10 Images

This module implements a Generative Adversarial Network (GAN) for detecting poisoned images within a subset of the CIFAR10 dataset.

Methodology:
- **Data Preparation:** Images are divided into 'poisoned' and 'clean' for training, and there's an option for an unlabeled dataset for prediction.
- **Model Architecture:**
  - **Generator:** Produces images to mimic either clean or poisoned samples based on a latent vector.
  - **Discriminator:** Trained to distinguish between real (either clean or poisoned) and fake images. It's used to detect anomalies by identifying images that don't fit the learned distribution.
- **Training:**
  - The Generator learns to produce images that could pass as poisoned, while the Discriminator learns to tell them apart from real images.
  - Checkpoints are saved to resume training if interrupted.
- **Prediction:** After training, the Discriminator can classify new images as potentially poisoned based on their likelihood of being 'real' or 'fake'.
- **Model Persistence:**
  - The model can be saved and loaded to avoid retraining.

This script includes:
- Training of the GAN model with checkpointing
- Utilities for loading models, resuming training, and making predictions on unlabeled data
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class CIFAR10LabeledDataset(Dataset):
    """Custom dataset to load labeled CIFAR10 images.

    Attributes:
        root_dir (str): Directory containing 'poisoned' and 'clean' subdirectories.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        images (list): List of paths to image files.
        labels (list): Corresponding labels for images, 1 for 'poisoned', 0 for 'clean'.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label in ['poisoned', 'clean']:
            label_dir = os.path.join(root_dir, label)
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                self.images.append(img_path)
                self.labels.append(1 if label == 'poisoned' else 0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class CIFAR10UnlabeledDataset(Dataset):
    """Custom dataset to load unlabeled CIFAR10 images for prediction.

    Attributes:
        root_dir (str): Directory containing images for prediction.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        images (list): Sorted list of paths to image files for prediction.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Sorting here ensures predictions are written in the order of file names
        self.images = sorted([os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class Generator(nn.Module):
    """Generator for producing images based on noise input.

    Args:
        latent_dim (int): Dimension of the latent space for noise input.
        channels_img (int): Number of channels in the output image (3 for RGB).
        features_g (int): Number of features in the first convolutional layer.
    """
    def __init__(self, latent_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, features_g * 4 * 4),
            nn.LeakyReLU(0.2)
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g * 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(features_g * 2, features_g, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(features_g, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = x.view(x.shape[0], -1, 4, 4)
        return self.model(x)

class Discriminator(nn.Module):
    """Discriminator for distinguishing between real and generated images.

    Args:
        channels_img (int): Number of channels in the input image.
        features_d (int): Number of features in the first convolutional layer.
    """
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(features_d * 4 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
latent_dim = 100  # Size of z latent vector (i.e., size of generator input)
channels_img = 3  # CIFAR-10 images are RGB
features_g = 64  # Starting feature size for generator
features_d = 64  # Starting feature size for discriminator
batch_size = 32  # Number of images in each mini-batch during training
epochs = 100  # Total number of epochs to train
checkpoint_dir = 'checkpoints'  # Directory to store model checkpoints
model_path = 'gan_model.pth'  # Path to save final model

# Data transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 images are 32x32, so resize if needed
    transforms.ToTensor(),         # Convert PIL Image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1] for better training stability
])

# Load dataset for training
dataset = CIFAR10LabeledDataset(root_dir='data', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Shuffle is set to True for training to ensure the model sees a variety of data, reducing overfitting

# Initialize models
generator = Generator(latent_dim, channels_img, features_g)
discriminator = Discriminator(channels_img, features_d)

# Loss function
criterion = nn.BCELoss()

# Optimizers
opt_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

def save_checkpoint(epoch, generator, discriminator, opt_g, opt_d):
    """Save checkpoint to resume training later.

    Args:
        epoch (int): Current training epoch
        generator (nn.Module): Generator model
        discriminator (nn.Module): Discriminator model
        opt_g (optim.Optimizer): Generator optimizer
        opt_d (optim.Optimizer): Discriminator optimizer
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # Naming the checkpoint with the epoch number for easy tracking
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'opt_g_state_dict': opt_g.state_dict(),
        'opt_d_state_dict': opt_d.state_dict(),
    }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

def load_checkpoint(generator, discriminator, opt_g, opt_d):
    """Load checkpoint to resume training.

    Args:
        generator (nn.Module): Generator model
        discriminator (nn.Module): Discriminator model
        opt_g (optim.Optimizer): Generator optimizer
        opt_d (optim.Optimizer): Discriminator optimizer

    Returns:
        int: The epoch to resume from
    """
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch')]
    if not checkpoints:
        return 0

    # Select the most recent checkpoint by parsing the epoch from the filename
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
    checkpoint = torch.load(os.path.join(checkpoint_dir, latest_checkpoint))

    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    opt_g.load_state_dict(checkpoint['opt_g_state_dict'])
    opt_d.load_state_dict(checkpoint['opt_d_state_dict'])

    return checkpoint['epoch'] + 1

def train(generator, discriminator, opt_g, opt_d, dataloader, epochs, resume_epoch=0):
    """Train GAN with checkpointing.

    Args:
        generator (nn.Module): Generator model
        discriminator (nn.Module): Discriminator model
        opt_g (optim.Optimizer): Generator optimizer
        opt_d (optim.Optimizer): Discriminator optimizer
        dataloader (DataLoader): DataLoader for training data
        epochs (int): Number of epochs to train
        resume_epoch (int): Epoch to start from if resuming

    This function trains the GAN, saving checkpoints after each epoch to allow for resuming training.
    """
    for epoch in range(resume_epoch, epochs):
        for i, (real_imgs, labels) in enumerate(dataloader):
            # Train Discriminator
            opt_d.zero_grad()

            # Real images
            real_labels = labels.float().view(-1, 1)
            real_output = discriminator(real_imgs)
            d_real_loss = criterion(real_output, real_labels)

            # Fake images
            noise = torch.randn(real_imgs.size(0), latent_dim)
            fake_imgs = generator(noise)
            fake_labels = torch.zeros(real_imgs.size(0), 1)
            fake_output = discriminator(fake_imgs.detach())
            d_fake_loss = criterion(fake_output, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            opt_d.step()

            # Train Generator
            opt_g.zero_grad()
            noise = torch.randn(real_imgs.size(0), latent_dim)
            fake_imgs = generator(noise)
            g_output = discriminator(fake_imgs)
            g_loss = criterion(g_output, torch.ones(real_imgs.size(0), 1))
            g_loss.backward()
            opt_g.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}] Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

        # Save checkpoint every epoch for potential resumption
        save_checkpoint(epoch, generator, discriminator, opt_g, opt_d)

    # Save final model after training
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
    }, model_path)

def load_model(generator, discriminator):
    """Load previously trained model if it exists.

    Returns:
        tuple: Generator and Discriminator models
    """
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        print("Model loaded from:", model_path)
        return generator, discriminator
    else:
        print("No saved model found, training from scratch.")
        return generator, discriminator

def export_predictions(model, dataloader, output_file):
    """Export predictions to 'labels.txt'. Predictions are 1 for poisoned, 0 for clean.

    Args:
        model (nn.Module): Trained discriminator model
        dataloader (DataLoader): DataLoader containing images to be classified
        output_file (str): Path where to save the predictions

    This function uses the discriminator to classify images, writing predictions in order to a file.
    """
    model.eval()
    with torch.no_grad():
        predictions = []
        for imgs in dataloader:
            outputs = model(imgs)
            # Thresholding: if probability > 0.5, classify as 'poisoned' (1), else 'clean' (0)
            predicted = (outputs > 0.5).float().squeeze().cpu().numpy()
            predictions.extend(predicted)

    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(f"{'bad' if int(pred) == 1 else 'good'}\n")

# Main execution
if __name__ == "__main__":
    # Check for existing model or resume from checkpoint
    resume_epoch = load_checkpoint(generator, discriminator, opt_g, opt_d)
    if resume_epoch == 0:
        generator, discriminator = load_model(generator, discriminator)
    else:
        print(f"Resuming training from epoch {resume_epoch}")

    # Training
    if resume_epoch < epochs:
        train(generator, discriminator, opt_g, opt_d, dataloader, epochs, resume_epoch)

    # Predictions
    unlabeled_dataset = CIFAR10UnlabeledDataset(root_dir='unlabeled_data', transform=transform)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)
    # Shuffle is set to False for prediction to maintain order of images

    export_predictions(discriminator, unlabeled_dataloader, 'labels.txt')

    print("Training completed and predictions exported to labels.txt")
