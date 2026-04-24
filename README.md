# HybridUNet — Medical Image Segmentation

Muhammad Haseeb (B23F0083AI116)  Hamza Saboor (B23F0066AI138)
AI-Red

## Project Files

| File | Purpose |
|------|---------|
| `model.py` | HybridSSM-UNet architecture (CNN + SSM + Cross-Attention fusion) |
| `dataset.py` | All dataset loaders + synthetic data generator |
| `train.py` | Training, evaluation, and result plots |
| `requirements.txt` | Dependencies |

## Install
pip install -r requirements.txt

## Step 1 — Test model
python model.py

## Step 2 — Run full pipeline on synthetic data

# Binary segmentation (like BUSI/ISIC)
python train.py --dataset synthetic_binary --epochs 20 --img_size 128

# CT multi-organ (like Synapse)
python train.py --dataset synthetic_ct --epochs 20 --img_size 128

# MRI brain tumor (like BraTS)
python train.py --dataset synthetic_mri --epochs 20 --img_size 128

## Step 3 — Run on real data (after downloading)

# BUSI ultrasound
python train.py --dataset busi --data_root data/BUSI --epochs 50

# ISIC 2018 skin lesion
python train.py --dataset isic --data_root data/ISIC --epochs 50

# Synapse CT (2D slices)
python train.py --dataset synapse --data_root data/Synapse --epochs 100 --in_ch 1

# BraTS 2023 MRI (2D slices)
python train.py --dataset brats --data_root data/BraTS --epochs 100

## Outputs

After training, the `outputs/` folder contains:

| File | Description |
|------|-------------|
| `best.pth` | Best model weights |
| `log.csv` | Per-epoch loss and DSC |
| `results.csv` | Final test metrics |
| `training_curves.png` | Loss and DSC over epochs |
| `predictions.png` | Input / Ground Truth / Prediction |
| `per_class_dsc.png` | Per-class Dice score chart |
| `comparison.png` | Our model vs baselines |
| `confusion_matrix.png` | Normalized confusion matrix |


## Evaluate saved model

python train.py --dataset synthetic_binary --eval_only --checkpoint outputs/best.pth

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | synthetic_binary | Dataset name |
| `--data_root` | None | Path to data (not needed for synthetic) |
| `--epochs` | 30 | Training epochs |
| `--batch_size` | 8 | Batch size |
| `--img_size` | 128 | Image size (use 256 for full quality) |
| `--base_ch` | 32 | Model width (16=smaller, 32=full) |
| `--out_dir` | results | Where to save results |
| `--workers` | 0 | DataLoader workers (0=safe on Windows) |
