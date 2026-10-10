# LiExNet FER Replication & Dataset Cartography

This repository contains the replication, optimization, and advanced training dynamics analysis of **LiExNet** for Facial Expression Recognition (FER) using the **RAF-7** dataset (7 emotion classes). 

---

## 🚀 Key Features & Highlights

1. **Lightweight Architecture (~0.6 MB Checkpoint, ~42k Parameters):**
   * Engineered with 4 optimized `LiExBlocks` combining Depthwise Separable Convolutions and Efficient Channel Attention (ECA).
   * Incorporates Global Average Pooling (`nn.AdaptiveAvgPool2d`) and $1 \times 1$ pointwise convolutions to resolve high-dimensional parameter explosions and ensure extreme edge-device efficiency.

2. **Dataset Cartography Integration:**
   * Automatically tracks learning dynamics for every individual training sample across all epochs ($E$).
   * Measures three critical statistical indicators:
     * **Confidence ($\hat{\mu}_i$):** Average true-class probability across epochs.
     * **Variability ($\hat{\sigma}_i$):** Standard deviation of true-class probability, capturing prediction stability.
     * **Correctness:** Proportion of epochs where the sample was correctly classified.
   * Automatically classifies data into three Data Map regions (**Easy-to-learn**, **Ambiguous**, and **Hard-to-learn**) to detect labeling errors and noise.
   * Exports an automated diagnostic report to `weights/dataset_cartography.csv` upon training completion.

---

## 📁 Project Structure

```text
liexnet-fer-replication/
├── data/
│   └── RAF_local_cleaned/          # Cleaned RAF-7 dataset (train/test splits)
├── models/
│   ├── liex_block.py               # LiExBlock definition (Depthwise Sep Conv + ECA)
│   └── liexnet.py                  # Main LiExNet architecture
├── utils/
│   └── dataset_loader.py           # Dataset loaders and transforms
├── weights/
│   ├── best_liexnet.pth            # Best validation checkpoint during training
│   ├── liexnet_deploy.pth          # Clean standalone deployment weights (~0.6 MB)
│   └── dataset_cartography.csv     # Exported sample-level training dynamics report
├── train.py                        # Full training script with Cartography tracking
└── README.md
```

---

## 📊 Dataset Cartography Data Map Regions

After training, the exported `dataset_cartography.csv` segments training images using median-split thresholds ($\tilde{\mu}, \tilde{\sigma}$):
* **Easy-to-learn ($\hat{\mu} \ge \tilde{\mu}, \hat{\sigma} < \tilde{\sigma}$):** Consistently recognized, clean data instances.
* **Ambiguous ($\hat{\sigma} \ge \tilde{\sigma}$):** High prediction volatility, representing borderline or complex expressions.
* **Hard-to-learn ($\hat{\mu} < \tilde{\mu}, \hat{\sigma} < \tilde{\sigma}$):** Consistently misclassified instances, frequently flagging potential **label noise** or annotation errors.

---

## ⚙️ Quick Start & Usage

### 1. Run Training with Dataset Cartography Tracking
Execute the training script to train the model and automatically generate both the model weights and the cartography CSV report:
```bash
python train.py --data_dir data/RAF_local_cleaned/train --epochs 200 --batch_size 32
```

### 2. Generate Deployment Checkpoint
Extract clean state dictionary weights for production or edge deployment:
```python
import torch
from models.liexnet import LiExNet

model = LiExNet(num_classes=7)
checkpoint = torch.load("weights/best_liexnet.pth", map_location="cpu")
model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
torch.save(model.state_dict(), "weights/liexnet_deploy.pth")