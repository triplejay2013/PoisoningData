"""
This is a simple sampler to test my generated poisoned data. I expect to see
my simple model to perform significantly worse with the poisoned data.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ---------------------------
# Dry Run Mode Toggle
# ---------------------------
# This variable allows you to switch to "dry run" mode, where training is done on a subset of the data for faster testing.
# In dry run mode, we'll reduce the dataset size and run fewer epochs for quick testing.

dry_run = True

# ---------------------------
# Data Loading
# ---------------------------

# Define a standard transform for CIFAR-10 and poisoned data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 clean dataset
trainset_clean = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader_clean = DataLoader(trainset_clean, batch_size=64, shuffle=True, num_workers=2)

# Custom Dataset for the poisoned data (local images) from your own folder
class PoisonedCIFAR10(Dataset):
    def __init__(self, poisons_dir, transform=None):
        self.poisons_dir = poisons_dir
        self.transform = transform

        # Get all class names (subdirectories) in poisons_dir
        self.classes = sorted(os.listdir(poisons_dir))

        # Create a list of (image_path, label) tuples
        self.image_paths = []
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(poisons_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append((img_path, label_idx))  # Use label_idx as the label (from class name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

# Transform for the poisoned images (optional resizing and normalization)
poisoned_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

poisons_dir = './poisons/train'
# Load the poisoned dataset
trainset_poisoned = PoisonedCIFAR10(poisons_dir, transform=poisoned_transform)

# Default: Combine full datasets
combined_data = torch.utils.data.ConcatDataset([trainset_clean, trainset_poisoned])
trainloader_poisoned = DataLoader(combined_data, batch_size=64, shuffle=True, num_workers=2)

# Dry run overrides
if dry_run:
    print("Dry run mode enabled: reducing dataset size for testing")

    # Define sample size for the dry run
    DRY_SAMPLE_SIZE = 50

    # Select only the first DRY_SAMPLE_SIZE samples from the clean and poisoned datasets
    clean_subset = torch.utils.data.Subset(trainset_clean, range(min(DRY_SAMPLE_SIZE, len(trainset_clean))))
    poisoned_subset = torch.utils.data.Subset(trainset_poisoned, range(min(DRY_SAMPLE_SIZE, len(trainset_poisoned))))

    # Combine subsets into a single dataset
    combined_data = torch.utils.data.ConcatDataset([clean_subset, poisoned_subset])

    # DataLoader for the dry run
    trainloader_poisoned = DataLoader(combined_data, batch_size=64, shuffle=True, num_workers=2)
    trainloader_clean = torch.utils.data.DataLoader(trainset_clean, batch_size=64, num_workers=2, sampler=torch.utils.data.SubsetRandomSampler(range(DRY_SAMPLE_SIZE * 2)))


    # Debugging: Ensure the combined dataset contains the expected number of samples
    print(f"Dry run: Combined dataset size = {len(combined_data)}")

# ---------------------------
# Define the Model (Simple CNN)
# ---------------------------

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define 3 convolution layers with increasing depth
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Fully connected layers after flattening the output
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # Pass input through layers with ReLU activations and max-pooling
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)  # Max pooling with stride 2
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 128 * 4 * 4)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output logits
        return x

# ---------------------------
# Training Function
# ---------------------------
def train_model(trainloader, model, criterion, optimizer, num_epochs=10):
    model.train()  # Set the model to training mode
    train_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            optimizer.zero_grad()  # Clear the gradients

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Compute the gradients
            optimizer.step()  # Update the weights

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        train_losses.append((epoch_loss, epoch_acc))
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    return train_losses

# ---------------------------
# Initialize Model, Loss, and Optimizer
# ---------------------------
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------
# Train on Clean Data
# ---------------------------
train_losses_clean = train_model(trainloader_clean, model, criterion, optimizer, num_epochs=5 if dry_run else 10)

# ---------------------------
# Reset Model for Poisoned Data
# ---------------------------
model = SimpleCNN()  # Reinitialize the model

# ---------------------------
# Train on Poisoned Data
# ---------------------------
train_losses_poisoned = train_model(trainloader_poisoned, model, criterion, optimizer, num_epochs=5 if dry_run else 10)

# ---------------------------
# Plot the Results
# ---------------------------
def plot_results(losses_clean, losses_poisoned):
    epochs = range(1, len(losses_clean) + 1)

    # Loss plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [x[0] for x in losses_clean], label='Clean Data Loss')
    plt.plot(epochs, [x[0] for x in losses_poisoned], label='Poisoned Data Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Comparison')

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [x[1] for x in losses_clean], label='Clean Data Accuracy')
    plt.plot(epochs, [x[1] for x in losses_poisoned], label='Poisoned Data Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training Accuracy Comparison')

    plt.tight_layout()
    # Save the plot to a file
    plt.savefig("output_plot.png", dpi=300, bbox_inches="tight")
    print("SANITY")

    # Clear the current plot to avoid overlaps in future plots
    plt.clf()

# Plot the training results (Loss and Accuracy)
plot_results(train_losses_clean, train_losses_poisoned)
