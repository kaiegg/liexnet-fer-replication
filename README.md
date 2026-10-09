# LiExNet FER Replication & Edge Deployment

This repository contains the PyTorch implementation, training pipeline, and edge-deployment evaluation for **LiExNet** (Lightweight Expression Recognition Network), a compact deep learning architecture designed for real-time Facial Emotion Recognition (FER)[cite: 4, 7].

---

## 📄 Model Overview

* **Parameter Count:** ~42,000 parameters[cite: 4, 7]
* **Model Memory Footprint:** ~0.17 MB[cite: 4, 16]
* **Peak Inference Memory:** ~2.0 MB[cite: 4, 16]
* **FLOPs:** 86 MFLOPs per sample ($128 \times 128$ input)[cite: 4, 16]
* **Target Throughput:** $>530$ FPS on NVIDIA Jetson embedded platforms (FP16 precision)[cite: 4, 16]

---

## 🗂️ Project Structure

```text
liexnet-fer-replication/
│
├── data/
│   └── RAF_Local/
│       ├── train/              # Subfolders: 0_Neutral .. 6_Anger
│       └── val/ (or test/)     # Validation image split
│
├── models/
│   ├── eca.py                  # Efficient Channel Attention (1D Conv)
│   ├── liex_block.py           # Conv2D + DwConv2D + ECA + MaxPool block
│   └── liexnet.py              # 4-block feature extractor + 2-layer FC head
│
├── utils/
│   └── dataset_loader.py       # PyTorch ImageFolder DataLoader pipeline
│
├── weights/                    # Checkpoints folder (auto-created during training)
│   └── best_liexnet.pth
│
├── train.py                    # Complete training, validation, and checkpointing loop
├── PROJECT_STATUS.md           # Milestone log and context memory
├── README.md                   # Project documentation
└── requirements.txt            # Dependency configuration
