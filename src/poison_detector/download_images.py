import os
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToPILImage

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def save_cifar10_images(img_class=9, data_folder="mixed_data/clean"):
    # Create the directory if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)

    # Download the CIFAR-10 dataset
    dataset = CIFAR10(root="data", train=True, download=True)

    # Filter and save truck images
    to_pil = ToPILImage()
    count = 0

    for idx, (image, label) in enumerate(zip(dataset.data, dataset.targets)):
        if label == img_class:
            count += 1
            img = to_pil(image)
            img.save(os.path.join(data_folder, f"{count:04d}.png"))

    print(f"Saved {count} truck images to {data_folder}.")

# Run the function
save_cifar10_images()
