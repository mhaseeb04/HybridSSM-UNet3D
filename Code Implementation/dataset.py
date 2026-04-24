"""
dataset.py
Dataset loading for all four benchmarks.

IMPORTANT: Real datasets require downloading first.
This file includes a SyntheticDataset that generates fake data automatically
so you can run train.py and test.py immediately to verify the full pipeline
works before downloading real data.

Real dataset folder structures expected:
  BUSI:
    data/BUSI/benign/   *.png  + *_mask.png
    data/BUSI/malignant/*.png  + *_mask.png

  ISIC:
    data/ISIC/ISIC2018_Task1-2_Training_Input/*.jpg
    data/ISIC/ISIC2018_Task1_Training_GroundTruth/*_segmentation.png

  Synapse CT (2D slices):
    data/Synapse/train/img/*.png   (axial slices)
    data/Synapse/train/label/*.png

  BraTS (2D slices from 3D MRI):
    data/BraTS/train/img/*.png     (4-channel stacked as RGBA or separate)
    data/BraTS/train/label/*.png
"""

import os
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def split_indices(n, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    t = int(0.70 * n)
    v = int(0.85 * n)
    return idx[:t], idx[t:v], idx[v:]


def basic_augment(image, mask, is_train):
    """Simple augmentation using numpy only — no external library needed."""
    if not is_train:
        return image, mask
    # Random horizontal flip
    if np.random.random() < 0.5:
        image = np.flip(image, axis=2).copy()
        mask  = np.flip(mask,  axis=1).copy()
    # Random vertical flip
    if np.random.random() < 0.3:
        image = np.flip(image, axis=1).copy()
        mask  = np.flip(mask,  axis=0).copy()
    # Random brightness shift
    if np.random.random() < 0.4:
        shift = np.random.uniform(-0.15, 0.15)
        image = np.clip(image + shift, 0, 1)
    return image, mask


# ---------------------------------------------------------------------------
# Synthetic dataset — runs immediately, no download needed
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """
    Generates random synthetic images and segmentation masks on the fly.
    Use this to test the full training pipeline before downloading real data.
    Each sample is a random image with a random circular/elliptical mask.

    Args:
        n_samples   : number of synthetic samples
        in_channels : 1 (grayscale), 3 (RGB), or 4 (MRI multi-modal)
        num_classes : number of segmentation classes
        img_size    : spatial size (square)
        is_train    : apply augmentation
    """
    def __init__(self, n_samples=200, in_channels=3, num_classes=2,
                 img_size=256, is_train=True):
        self.n        = n_samples
        self.in_ch    = in_channels
        self.n_cls    = num_classes
        self.size     = img_size
        self.is_train = is_train
        np.random.seed(42 if not is_train else 0)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        H = W = self.size
        rng = np.random.default_rng(idx)

        # Generate synthetic image
        image = rng.uniform(0.1, 0.9, (self.in_ch, H, W)).astype(np.float32)

        # Generate synthetic mask with 1-3 random ellipses
        mask = np.zeros((H, W), dtype=np.int64)
        n_objects = rng.integers(1, min(self.n_cls, 4))
        yy, xx   = np.mgrid[:H, :W]
        for cls in range(1, n_objects + 1):
            cy = rng.integers(H // 4, 3 * H // 4)
            cx = rng.integers(W // 4, 3 * W // 4)
            ry = rng.integers(H // 8, H // 4)
            rx = rng.integers(W // 8, W // 4)
            ellipse = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
            mask[ellipse] = min(cls, self.n_cls - 1)

        # Simulate texture inside mask
        n_fg = int((mask > 0).sum())
        for c in range(self.in_ch):
            if n_fg > 0:
                noise = rng.uniform(0.1, 0.3, n_fg).astype(np.float32)
                flat = image[c].copy()
                flat[mask > 0] = np.clip(flat[mask > 0] + noise, 0, 1)
                image[c] = flat
        image = np.clip(image, 0, 1)

        if self.is_train:
            image, mask = basic_augment(image, mask, True)

        return (torch.tensor(image, dtype=torch.float32),
                torch.tensor(mask,  dtype=torch.long))


# ---------------------------------------------------------------------------
# BUSI dataset
# ---------------------------------------------------------------------------

class BUSIDataset(Dataset):
    """
    BUSI breast ultrasound dataset.
    Expects: data_root/{benign,malignant}/*.png and *_mask.png
    """
    def __init__(self, data_root, split="train", img_size=256):
        pairs = []
        for cls in ("benign", "malignant"):
            cls_dir = os.path.join(data_root, cls)
            if not os.path.exists(cls_dir):
                continue
            for p in sorted(glob.glob(os.path.join(cls_dir, "*.png"))):
                if "_mask" in p:
                    continue
                m = p.replace(".png", "_mask.png")
                if os.path.exists(m):
                    pairs.append((p, m))
        if not pairs:
            raise FileNotFoundError(f"No BUSI images found in {data_root}")

        tr, va, te = split_indices(len(pairs))
        sel = {"train": tr, "val": va, "test": te}[split]
        self.pairs    = [pairs[i] for i in sel]
        self.is_train = (split == "train")
        self.size     = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img  = np.array(Image.open(ip).convert("RGB").resize(
                         (self.size, self.size))) / 255.0            # (H,W,3)
        mask = np.array(Image.open(mp).convert("L").resize(
                         (self.size, self.size), Image.NEAREST))
        mask = (mask > 127).astype(np.int64)
        image = img.transpose(2, 0, 1).astype(np.float32)           # (3,H,W)
        image, mask = basic_augment(image, mask, self.is_train)
        return torch.tensor(image), torch.tensor(mask)


# ---------------------------------------------------------------------------
# ISIC 2018 dataset
# ---------------------------------------------------------------------------

class ISICDataset(Dataset):
    """
    ISIC 2018 skin lesion segmentation.
    Expects:
      data_root/ISIC2018_Task1-2_Training_Input/*.jpg
      data_root/ISIC2018_Task1_Training_GroundTruth/*_segmentation.png
    """
    def __init__(self, data_root, split="train", img_size=256):
        img_dir  = os.path.join(data_root, "ISIC2018_Task1-2_Training_Input")
        mask_dir = os.path.join(data_root, "ISIC2018_Task1_Training_GroundTruth")
        pairs = []
        for ip in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
            base = os.path.splitext(os.path.basename(ip))[0]
            mp   = os.path.join(mask_dir, f"{base}_segmentation.png")
            if os.path.exists(mp):
                pairs.append((ip, mp))
        if not pairs:
            raise FileNotFoundError(f"No ISIC images found in {data_root}")

        tr, va, te = split_indices(len(pairs))
        sel = {"train": tr, "val": va, "test": te}[split]
        self.pairs    = [pairs[i] for i in sel]
        self.is_train = (split == "train")
        self.size     = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img  = np.array(Image.open(ip).convert("RGB").resize(
                         (self.size, self.size))) / 255.0
        mask = np.array(Image.open(mp).convert("L").resize(
                         (self.size, self.size), Image.NEAREST))
        mask  = (mask > 127).astype(np.int64)
        image = img.transpose(2, 0, 1).astype(np.float32)
        image, mask = basic_augment(image, mask, self.is_train)
        return torch.tensor(image), torch.tensor(mask)


# ---------------------------------------------------------------------------
# Generic 2D slice dataset (Synapse CT / BraTS 2D slices)
# ---------------------------------------------------------------------------

class SliceDataset(Dataset):
    """
    Generic dataset for 2D PNG slices pre-extracted from 3D volumes.
    img_dir  : folder with input PNG files
    mask_dir : folder with label PNG files (same filename)
    in_channels: 1 for CT grayscale, 3 or 4 for MRI (stored as RGB/RGBA)
    """
    def __init__(self, img_dir, mask_dir, split="train",
                 in_channels=1, img_size=256, num_classes=9):
        imgs  = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        if not imgs:
            raise FileNotFoundError(f"No PNG files in {img_dir}")
        pairs = list(zip(imgs, masks))

        tr, va, te = split_indices(len(pairs))
        sel = {"train": tr, "val": va, "test": te}[split]
        self.pairs      = [pairs[i] for i in sel]
        self.is_train   = (split == "train")
        self.in_ch      = in_channels
        self.size       = img_size
        self.num_classes= num_classes

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        mode  = "L" if self.in_ch == 1 else "RGB"
        img   = np.array(Image.open(ip).convert(mode).resize(
                          (self.size, self.size))) / 255.0
        mask  = np.array(Image.open(mp).convert("L").resize(
                          (self.size, self.size), Image.NEAREST)).astype(np.int64)
        mask  = np.clip(mask, 0, self.num_classes - 1)

        if self.in_ch == 1:
            image = img[np.newaxis].astype(np.float32)
        else:
            image = img.transpose(2, 0, 1).astype(np.float32)

        image, mask = basic_augment(image, mask, self.is_train)
        return torch.tensor(image), torch.tensor(mask)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

DATASET_INFO = {
    "synthetic_binary": dict(in_ch=3,  classes=2,  name="Synthetic Binary"),
    "synthetic_ct":     dict(in_ch=1,  classes=9,  name="Synthetic CT"),
    "synthetic_mri":    dict(in_ch=4,  classes=4,  name="Synthetic MRI"),
    "busi":             dict(in_ch=3,  classes=2,  name="BUSI Ultrasound"),
    "isic":             dict(in_ch=3,  classes=2,  name="ISIC 2018"),
    "synapse":          dict(in_ch=1,  classes=9,  name="Synapse CT"),
    "brats":            dict(in_ch=4,  classes=4,  name="BraTS 2023"),
}


def get_loaders(dataset, data_root=None, batch_size=8,
                img_size=256, num_workers=0):
    """
    Returns (train_loader, val_loader, test_loader, in_channels, num_classes).

    For synthetic datasets, data_root is ignored.
    For real datasets, data_root must point to the data folder.
    num_workers=0 is default (safe on Windows).
    """
    info = DATASET_INFO[dataset]
    in_ch  = info["in_ch"]
    n_cls  = info["classes"]

    if dataset.startswith("synthetic"):
        n = 300
        train_ds = SyntheticDataset(int(n*0.70), in_ch, n_cls, img_size, True)
        val_ds   = SyntheticDataset(int(n*0.15), in_ch, n_cls, img_size, False)
        test_ds  = SyntheticDataset(int(n*0.15), in_ch, n_cls, img_size, False)

    elif dataset == "busi":
        train_ds = BUSIDataset(data_root, "train", img_size)
        val_ds   = BUSIDataset(data_root, "val",   img_size)
        test_ds  = BUSIDataset(data_root, "test",  img_size)

    elif dataset == "isic":
        train_ds = ISICDataset(data_root, "train", img_size)
        val_ds   = ISICDataset(data_root, "val",   img_size)
        test_ds  = ISICDataset(data_root, "test",  img_size)

    elif dataset == "synapse":
        img_dir  = os.path.join(data_root, "train", "img")
        mask_dir = os.path.join(data_root, "train", "label")
        train_ds = SliceDataset(img_dir, mask_dir, "train", 1, img_size, 9)
        val_ds   = SliceDataset(img_dir, mask_dir, "val",   1, img_size, 9)
        test_ds  = SliceDataset(img_dir, mask_dir, "test",  1, img_size, 9)

    elif dataset == "brats":
        img_dir  = os.path.join(data_root, "train", "img")
        mask_dir = os.path.join(data_root, "train", "label")
        train_ds = SliceDataset(img_dir, mask_dir, "train", 4, img_size, 4)
        val_ds   = SliceDataset(img_dir, mask_dir, "val",   4, img_size, 4)
        test_ds  = SliceDataset(img_dir, mask_dir, "test",  4, img_size, 4)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    print(f"Dataset : {info['name']}")
    print(f"  Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")
    print(f"  in_channels={in_ch}  num_classes={n_cls}  img_size={img_size}")
    return train_loader, val_loader, test_loader, in_ch, n_cls
