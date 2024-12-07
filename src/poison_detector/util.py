import torch
import os

import random

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
import matplotlib.pyplot as plt

DEBUG = os.environ.get("DEBUG", False) in [True, "True", "true", "t", 1, "1"]


def load_data(batch_size=64):
    """
    Loads CIFAR10 and poisoned data, combines them, and returns a balanced DataLoader.

    Args:
        batch_size (int): Number of images per batch.

    Returns:
        DataLoader: DataLoader for the combined dataset.
    """

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to match CIFAR10 size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 normalization
    ])

    # Load the CIFAR10 dataset
    cifar10_dataset = datasets.CIFAR10(root="src/poison_detector/data", train=True, download=True, transform=transform)

    # Load the poisoned dataset using ImageFolder
    poisoned_dataset = datasets.ImageFolder(root="src/poison_detector/poison", transform=transform)

    # If DEBUG is enabled, scale down both datasets for faster testing
    if DEBUG:
        print("DEBUG mode is enabled. Scaling down the datasets.")
        scale_percentage = 0.05

        # Scale down CIFAR10
        num_samples_cifar10 = len(cifar10_dataset)
        scaled_num_samples_cifar10 = int(num_samples_cifar10 * scale_percentage)
        cifar10_indices = random.sample(range(num_samples_cifar10), scaled_num_samples_cifar10)
        cifar10_dataset = Subset(cifar10_dataset, cifar10_indices)

        # Scale down poisoned data
        num_samples_poisoned = len(poisoned_dataset)
        scaled_num_samples_poisoned = int(num_samples_poisoned * scale_percentage)
        poisoned_indices = random.sample(range(num_samples_poisoned), scaled_num_samples_poisoned)
        poisoned_dataset = Subset(poisoned_dataset, poisoned_indices)

        print(f"Scaled down CIFAR10 dataset to {scaled_num_samples_cifar10} samples.")
        print(f"Scaled down poisoned dataset to {scaled_num_samples_poisoned} samples.")

    # Combine CIFAR10 and poisoned datasets
    combined_dataset = ConcatDataset([cifar10_dataset, poisoned_dataset])

    # Create DataLoader for batching the images
    poisoned_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    return poisoned_loader

def plot_samples(data_loader, n_samples=16):
    """
    Plot random samples from the dataset.

    Args:
        data_loader (DataLoader): DataLoader containing dataset.
        n_samples (int): Number of samples to display.
    """
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples, 2))
    for i in range(n_samples):
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Unnormalize
        ax.axis('off')
    plt.show()

def generate_labels(data_loader, model, output_path="labels.txt"):
    """
    Generates labels for each image in the dataset, indicating whether it's poisoned or not.
    The labels will be saved in a text file as 'good' or 'bad'.

    Args:
        data_loader (DataLoader): DataLoader for the dataset (CIFAR10 + poisoned).
        model (torch.nn.Module): The trained model (e.g., a GAN discriminator or classifier).
        output_path (str): Path to save the generated labels.
    """
    model.eval()

    labels = []
    filenames = []

    with torch.no_grad():
        for images, _ in data_loader:
            # Forward pass through the model
            outputs = model(images)

            # Assuming the model outputs logits or probabilities for "good" (0) or "bad" (1)
            predictions = torch.argmax(outputs, dim=1)  # For binary classification: 0 = good, 1 = bad

            # Add labels and corresponding filenames
            for idx, prediction in enumerate(predictions):
                label = "bad" if prediction == 1 else "good"  # Assuming 1=poisoned, 0=clean
                labels.append(label)

                # Retrieve the filenames for the images in the batch
                if hasattr(data_loader.dataset, "samples"):
                    filename = data_loader.dataset.samples[idx][0]  # Assuming (file_path, label) structure
                elif isinstance(data_loader.dataset, torch.utils.data.Subset):
                    filename = data_loader.dataset.dataset.samples[idx][0]  # Assuming (file_path, label) structure
                else:
                    filename = None
                filenames.append(filename)

    # Sort filenames and labels based on filenames
    sorted_labels_filenames = sorted(zip(filenames, labels), key=lambda x: x[0])

    # Write the sorted labels to a text file
    with open(output_path, "w") as file:
        for filename, label in sorted_labels_filenames:
            file.write(f"{label}\n")
