import os
import argparse
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from models.liexnet import LiExNet
from utils.dataset_loader import get_dataloaders


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="LiExNet FER Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="data/RAF_Local", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--img_size", type=int, default=128, help="Image resolution")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory to save model weights")
    args = parser.parse_args()

    # Hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting LiExNet Training Pipeline ===")
    print(f"Using Device: {device}")

    # Ensure weights directory exists
    os.makedirs(args.weights_dir, exist_ok=True)

    # DataLoaders
    train_loader, val_loader, classes = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    num_classes = len(classes)
    print(f"Dataset Loaded | Classes ({num_classes}): {classes}")

    # Model, Loss, Optimizer & Scheduler
    model = LiExNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    best_val_acc = 0.0
    best_weights_path = os.path.join(args.weights_dir, "best_liexnet.pth")

    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Learning rate adjustment based on validation loss
        scheduler.step(val_loss)

        # Checkpoint saving
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'classes': classes
            }, best_weights_path)

        # Print Epoch Progress
        print(f"Epoch [{epoch+1:03d}/{args.epochs:03d}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}% "
              f"{'★ Saved Best' if is_best else ''}")

    total_time = time.time() - start_time
    print(f"\n=== Training Complete ===")
    print(f"Total Time: {total_time/60:.2f} mins")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Saved Checkpoint: {best_weights_path}")


if __name__ == "__main__":
    main()
