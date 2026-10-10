#Training with saved metadata in csv file

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.liexnet import LiExNet
from utils.dataset_loader import get_dataloaders


class IndexedDataset(torch.utils.data.Dataset):
  """Wrapper to append sample index to dataset outputs for Cartography tracking."""

  def __init__(self, dataset):
    self.dataset = dataset

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    img, label = self.dataset[idx]
    return img, label, idx


def train_one_epoch(
    model, dataloader, criterion, optimizer, device, epoch, prob_history
):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for images, labels, indices in dataloader:
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Track true class probabilities for Dataset Cartography
    with torch.no_grad():
      probs = torch.softmax(outputs, dim=1)
      true_class_probs = (
          probs[torch.arange(labels.size(0), device=device), labels]
          .detach()
          .cpu()
          .numpy()
      )
      prob_history[indices.numpy(), epoch] = true_class_probs

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
  parser.add_argument(
      "--data_dir",
      type=str,
      default="data/RAF_Local",
      help="Path to dataset directory",
  )
  parser.add_argument(
      "--epochs", type=int, default=200, help="Number of training epochs"
  )
  parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
  parser.add_argument(
      "--lr", type=float, default=1e-3, help="Initial learning rate"
  )
  parser.add_argument(
      "--img_size", type=int, default=128, help="Image resolution"
  )
  parser.add_argument(
      "--weights_dir",
      type=str,
      default="weights",
      help="Directory to save model weights",
  )
  args = parser.parse_args()

  # Hardware setup
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("=== Starting LiExNet Training Pipeline ===")
  print(f"Using Device: {device}")

  # Ensure weights directory exists
  os.makedirs(args.weights_dir, exist_ok=True)

  # DataLoaders
  train_loader_raw, val_loader, classes = get_dataloaders(
      data_dir=args.data_dir, batch_size=args.batch_size, img_size=args.img_size
  )

  # Wrap train dataset to yield indices for Cartography
  indexed_train_dataset = IndexedDataset(train_loader_raw.dataset)
  train_loader = torch.utils.data.DataLoader(
      indexed_train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=2,
      drop_last=False,
  )

  num_classes = len(classes)
  print(f"Dataset Loaded | Classes ({num_classes}): {classes}")

  # Initialize tracking matrix for Dataset Cartography [num_samples, epochs]
  num_train_samples = len(train_loader_raw.dataset)
  prob_history = np.zeros((num_train_samples, args.epochs))

  # Model, Loss, Optimizer & Scheduler
  model = LiExNet(num_classes=num_classes).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
  scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

  best_val_acc = 0.0
  best_weights_path = os.path.join(args.weights_dir, "best_liexnet.pth")

  start_time = time.time()

  for epoch in range(args.epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch, prob_history
    )
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # Learning rate adjustment based on validation loss
    scheduler.step(val_loss)

    # Checkpoint saving
    is_best = val_acc > best_val_acc
    if is_best:
      best_val_acc = val_acc
      torch.save(
          {
              "epoch": epoch + 1,
              "model_state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "val_acc": val_acc,
              "val_loss": val_loss,
              "classes": classes,
          },
          best_weights_path,
      )

    # Print Epoch Progress
    print(
        f"Epoch [{epoch+1:03d}/{args.epochs:03d}] | "
        f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
        f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}% "
        f"{'★ Saved Best' if is_best else ''}"
    )

  # === Post-Training: Dataset Cartography Calculation & CSV Export ===
  print("\n=== Generating Dataset Cartography Analysis ===")
  confidences = np.mean(prob_history, axis=1)
  variability = np.std(prob_history, axis=1)

  # Median splits for Data Map region thresholds
  conf_median = np.median(confidences)
  var_median = np.median(variability)

  regions = []
  for conf, var in zip(confidences, variability):
    if conf >= conf_median and var < var_median:
      regions.append("Easy-to-learn")
    elif var >= var_median:
      regions.append("Ambiguous")
    else:
      regions.append("Hard-to-learn")

  # Extract file metadata if available from ImageFolder
  dataset_samples = getattr(train_loader_raw.dataset, "samples", None)
  if dataset_samples is not None:
    image_paths = [s[0] for s in dataset_samples]
    true_labels = [s[1] for s in dataset_samples]
  else:
    image_paths = [f"sample_{i}" for i in range(num_train_samples)]
    true_labels = [train_loader_raw.dataset[i][1] for i in range(num_train_samples)]

  cartography_df = pd.DataFrame({
      "sample_index": range(num_train_samples),
      "image_path": image_paths,
      "true_label": true_labels,
      "confidence": confidences,
      "variability": variability,
      "data_map_region": regions,
  })

  csv_path = os.path.join(args.weights_dir, "dataset_cartography.csv")
  cartography_df.to_csv(csv_path, index=False)

  total_time = time.time() - start_time
  print("=== Training & Cartography Complete ===")
  print(f"Total Time: {total_time/60:.2f} mins")
  print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
  print(f"Saved Checkpoint: {best_weights_path}")
  print(f"Saved Cartography Report: {csv_path}")


if __name__ == "__main__":
  main()
