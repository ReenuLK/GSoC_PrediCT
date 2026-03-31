# Project PrediCT: High-Throughput Cardiac Segmentation for CAC Scoring

## Overview
Project **PrediCT** is an ultra-high-throughput, memory-efficient cardiac segmentation pipeline designed for population-scale research.

While current State-of-the-Art (SOTA) tools often require **~60 seconds per volume**, this pipeline achieves a **~6000× speedup**, delivering **sub-second inference (~9.8 ms)** on commodity hardware such as the NVIDIA T4.

The system is built to automate the quantification of **Coronary Artery Calcium (CAC)**, a critical biomarker for predicting **Major Adverse Cardiac Events (MACE)**.

---

## Key Features

- **SegResNet-VAE Architecture**  
  Utilizes a SegResNet backbone with Variational Autoencoder (VAE) regularization to extract robust anatomical features and reduce overfitting.

- **Standardized Preprocessing**  
  Includes:
  - Isotropic resampling (1.0 mm)
  - Hounsfield Unit (HU) windowing [-160, 240]
  - Strict RAS orientation

- **High Throughput**  
  Achieves **107.25 volumes per second** with a compact **18 MB model footprint**.

---

## Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ReenuLK/GSoC_PrediCT.git
cd GSoC_PrediCT
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run Inference
```bash
notebooks/ReenuLK_PrediCT_Task-1.ipynb
```
