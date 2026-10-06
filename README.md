# LiExNet FER Replication & Edge Deployment

This repository contains the complete PyTorch implementation, training pipeline, and fine-tuning workflow for **LiExNet** (a lightweight deep learning architecture designed for real-time Facial Expression Recognition).

---

## 📄 Project Overview & Status

* **Objective:** Replicate LiExNet and fine-tune its 4-block architecture on standard facial expression datasets (e.g., RAF-DB, FER2013) for real-time edge deployment (e.g., NVIDIA Jetson / embedded devices)[cite: 35].
* **Execution Target:** Google Colab (T4 / V100 GPU) for cloud training, validation, and benchmarking.
* **Repository Strategy:** Direct clone into Colab’s local high-speed storage (`/content/`) for maximum image loading I/O throughput.

---

## 🗂️ Repository Layout

```text
liexnet-fer-replication/
│
├── data/
│   └── RAF_Local/
│       ├── train/          # Class subfolders: 0_Neutral .. 6_Anger
│       └── test/           # Validation/Test image split
│
├── models/
│   ├── __init__.py
│   ├── eca.py              # Efficient Channel Attention (1D adaptive Conv)
│   ├── liex_block.py       # Hierarchical Conv + Depthwise + ECA block
│   └── liexnet.py          # Full 4-block architecture stack & classifier
│
├── utils/
│   ├── __init__.py
│   └── dataset_loader.py   # PyTorch ImageFolder DataLoader pipeline
│
├── PROJECT_STATUS.md       # Detailed milestone tracking and context memory
├── README.md               # Quickstart and architecture summary
└── requirements.txt        # Package dependencies
