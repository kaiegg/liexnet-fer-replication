import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32, img_size=128, num_workers=2):
    """
    Constructs PyTorch DataLoaders for RAF_Local image directories.
    Applies image resizing (128x128), horizontal flips for augmentation,
    and converts images to float tensors in [0, 1].
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_path = os.path.join(data_dir, 'train')

    # Check if 'val' exists; if not, fall back to 'test'
    if os.path.exists(os.path.join(data_dir, 'val')):
        val_path = os.path.join(data_dir, 'val')
    elif os.path.exists(os.path.join(data_dir, 'test')):
        val_path = os.path.join(data_dir, 'test')
    else:
        val_path = train_path

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset.classes
