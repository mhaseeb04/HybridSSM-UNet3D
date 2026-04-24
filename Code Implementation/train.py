"""
train.py
Complete training, validation, and evaluation script.

Usage examples:

  # Run on synthetic data (no download needed — great for testing)
  python train.py --dataset synthetic_binary --epochs 10

  # Run on real BUSI data
  python train.py --dataset busi --data_root data/BUSI --epochs 50

  # Evaluate a saved model
  python train.py --dataset synthetic_binary --eval_only --checkpoint outputs/best.pth

All results (metrics, plots, checkpoint) are saved to --out_dir (default: outputs/).
"""

import os
import csv
import math
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import build_model
from dataset import get_loaders


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5):
        super().__init__()
        self.C     = num_classes
        self.smooth= smooth

    def forward(self, logits, targets):
        probs  = F.softmax(logits, dim=1)
        ndim   = targets.ndim
        oh     = F.one_hot(targets, self.C).permute(0, ndim, *range(1, ndim)).float()
        p      = probs.reshape(probs.shape[0], self.C, -1)
        g      = oh.reshape(oh.shape[0],   self.C, -1)
        inter  = (p * g).sum(-1)
        denom  = p.sum(-1) + g.sum(-1)
        dice   = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class CompoundLoss(nn.Module):
    """0.5 * Dice + 0.5 * CrossEntropy"""
    def __init__(self, num_classes):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce   = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return 0.5 * self.dice(logits, targets) + 0.5 * self.ce(logits, targets)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_np, true_np, num_classes):
    """
    Compute per-class DSC and IoU, return mean over foreground classes.
    pred_np, true_np: numpy int arrays of same shape
    """
    smooth = 1e-5
    dscs, ious = [], []
    for c in range(1, num_classes):
        p = (pred_np == c)
        t = (true_np == c)
        inter  = (p & t).sum()
        dsc    = (2 * inter + smooth) / (p.sum() + t.sum() + smooth)
        iou    = (inter + smooth) / ((p | t).sum() + smooth)
        dscs.append(float(dsc))
        ious.append(float(iou))
    return float(np.mean(dscs)), float(np.mean(ious))


def compute_precision_recall_f1(pred_np, true_np, num_classes):
    """Binary or multi-class precision, recall, F1 (macro average)."""
    smooth = 1e-5
    precs, recs, f1s = [], [], []
    for c in range(1, num_classes):
        p = (pred_np == c)
        t = (true_np == c)
        tp = (p & t).sum()
        fp = (p & ~t).sum()
        fn = (~p & t).sum()
        prec = (tp + smooth) / (tp + fp + smooth)
        rec  = (tp + smooth) / (tp + fn + smooth)
        f1   = 2 * prec * rec / (prec + rec + smooth)
        precs.append(float(prec))
        recs.append(float(rec))
        f1s.append(float(f1))
    return float(np.mean(precs)), float(np.mean(recs)), float(np.mean(f1s))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, num_classes, device):
    model.eval()
    total_loss = 0.0
    all_dsc, all_iou = [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        trues = labels.cpu().numpy()
        for p, t in zip(preds, trues):
            d, i = compute_metrics(p, t, num_classes)
            all_dsc.append(d)
            all_iou.append(i)
    return (total_loss / len(loader),
            float(np.mean(all_dsc)),
            float(np.mean(all_iou)))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_curves(log_path, out_dir):
    epochs, tr_loss, val_loss, val_dsc = [], [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            tr_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_dsc.append(float(row["val_dsc"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, tr_loss,  label="Train Loss", color="#2E74B5", lw=2)
    ax1.plot(epochs, val_loss, label="Val Loss",   color="#C05900", lw=2, ls="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_dsc, label="Val DSC", color="#217A3C", lw=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Dice Score")
    ax2.set_title("Validation Dice Score", fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


def plot_comparison(results, out_dir):
    """Bar chart: our model vs published baselines."""
    baselines = {
        "U-Net":        0.7912,
        "TransUNet":    0.7748,
        "Swin-UNet":    0.7913,
        "nnU-Net":      0.8701,
        "CSWin-UNet":   0.9158,
        "Ours":         results["mean_dsc"],
    }
    names  = list(baselines.keys())
    scores = [v * 100 for v in baselines.values()]
    colors = ["#B0B0B0"] * (len(names) - 1) + ["#1F3864"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, scores, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean DSC (%)", fontsize=11)
    ax.set_title("Comparison with Baseline Methods", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


def plot_predictions(model, loader, num_classes, device, out_dir, n=4):
    """Save a grid of input / ground-truth / prediction for n samples."""
    model.eval()
    images_shown, preds_shown, labels_shown = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds  = logits.argmax(dim=1).cpu().numpy()
            for i in range(len(images)):
                if len(images_shown) >= n:
                    break
                img = images[i].cpu().numpy()
                # Show first channel for display
                images_shown.append(img[0] if img.shape[0] >= 1 else img)
                preds_shown.append(preds[i])
                labels_shown.append(labels[i].numpy())
            if len(images_shown) >= n:
                break

    try:
        cmap = plt.colormaps.get_cmap("tab10").resampled(num_classes)
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10", num_classes)
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 2.5))
    if n == 1:
        axes = axes[np.newaxis]
    for i in range(n):
        axes[i, 0].imshow(images_shown[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(labels_shown[i], cmap=cmap,
                          vmin=0, vmax=num_classes - 1, interpolation="nearest")
        axes[i, 2].imshow(preds_shown[i],  cmap=cmap,
                          vmin=0, vmax=num_classes - 1, interpolation="nearest")
        if i == 0:
            axes[i, 0].set_title("Input",        fontsize=10, fontweight="bold")
            axes[i, 1].set_title("Ground Truth",  fontsize=10, fontweight="bold")
            axes[i, 2].set_title("Prediction",    fontsize=10, fontweight="bold")
        for ax in axes[i]:
            ax.axis("off")
    plt.suptitle("Sample Predictions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


def plot_per_class(per_class_dsc, out_dir):
    names  = list(per_class_dsc.keys())
    scores = [per_class_dsc[n] * 100 for n in names]
    colors = ["#4472C4" if s >= 80 else "#F4A261" for s in scores]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.5)))
    bars = ax.barh(names, scores, color=colors, edgecolor="white")
    for bar, val in zip(bars, scores):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("DSC (%)", fontsize=11)
    ax.set_title("Per-Class Dice Score", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.axvline(80, color="red", ls="--", lw=1, alpha=0.5, label="80% threshold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "per_class_dsc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


def plot_confusion(pred_all, true_all, num_classes, out_dir):
    from sklearn.metrics import confusion_matrix
    import numpy as np

    cm = confusion_matrix(true_all.flatten(), pred_all.flatten(),
                          labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(max(4, num_classes), max(4, num_classes)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True",      fontsize=10)
    ax.set_title("Normalized Confusion Matrix", fontsize=12, fontweight="bold")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, path):
        self.path = path
        self.t0   = time.time()
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_loss", "val_dsc", "val_iou", "lr"])

    def log(self, epoch, tl, vl, vd, vi, lr):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{tl:.5f}", f"{vl:.5f}",
                 f"{vd:.4f}", f"{vi:.4f}", f"{lr:.6f}"])
        elapsed = (time.time() - self.t0) / 60
        print(f"[{epoch:3d}]  loss={tl:.4f}  val_loss={vl:.4f}  "
              f"DSC={vd:.4f}  IoU={vi:.4f}  lr={lr:.2e}  {elapsed:.1f}min")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="synthetic_binary",
                        help="Dataset name. Use 'synthetic_binary', 'synthetic_ct', "
                             "'synthetic_mri', 'busi', 'isic', 'synapse', or 'brats'")
    parser.add_argument("--data_root",  default=None,
                        help="Path to dataset folder (not needed for synthetic)")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--img_size",   type=int,   default=128,
                        help="Image size (use 128 for quick test, 256 for full run)")
    parser.add_argument("--base_ch",    type=int,   default=32,
                        help="Base channels in model (32=full, 16=lightweight)")
    parser.add_argument("--workers",    type=int,   default=0,
                        help="DataLoader workers (0=safe on Windows)")
    parser.add_argument("--out_dir",    default="outputs")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eval_only",  action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Data
    train_loader, val_loader, test_loader, in_ch, n_cls = get_loaders(
        args.dataset, args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.workers,
    )

    # Model
    model = build_model(in_channels=in_ch, num_classes=n_cls,
                        base_ch=args.base_ch).to(device)
    print(f"Parameters: {model.num_params / 1e6:.2f}M")

    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")

    criterion = CompoundLoss(n_cls).to(device)
    ckpt_path = os.path.join(args.out_dir, "best.pth")
    log_path  = os.path.join(args.out_dir, "log.csv")

    # --------------- Training ---------------
    if not args.eval_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        use_amp   = device.type == "cuda"
        scaler    = torch.cuda.amp.GradScaler() if use_amp else None
        logger    = Logger(log_path)
        best_dsc  = 0.0
        patience  = 0

        for epoch in range(1, args.epochs + 1):
            tl = train_one_epoch(model, train_loader, criterion,
                                  optimizer, device, scaler)
            vl, vd, vi = evaluate(model, val_loader, criterion, n_cls, device)
            logger.log(epoch, tl, vl, vd, vi, scheduler.get_last_lr()[0])
            scheduler.step()

            if vd > best_dsc:
                best_dsc = vd
                patience = 0
                torch.save(model.state_dict(), ckpt_path)
                print(f"  Saved best checkpoint  (DSC={best_dsc:.4f})")
            else:
                patience += 1
                if patience >= 15:
                    print(f"Early stopping at epoch {epoch}")
                    break

        plot_curves(log_path, args.out_dir)
        # Load best for evaluation
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # --------------- Evaluation on test set ---------------
    print("\n--- Test Set Evaluation ---")
    all_preds, all_trues = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_trues.append(labels.numpy())

    preds_all = np.concatenate(all_preds, axis=0)
    trues_all = np.concatenate(all_trues, axis=0)

    # Overall metrics
    mean_dsc, mean_iou = compute_metrics(preds_all, trues_all, n_cls)
    prec, rec, f1      = compute_precision_recall_f1(preds_all, trues_all, n_cls)

    print(f"\n  Mean DSC       : {mean_dsc*100:.2f}%")
    print(f"  Mean IoU       : {mean_iou*100:.2f}%")
    print(f"  Precision      : {prec*100:.2f}%")
    print(f"  Recall         : {rec*100:.2f}%")
    print(f"  F1 Score       : {f1*100:.2f}%")

    # Per-class DSC
    per_class_dsc = {}
    from dataset import DATASET_INFO
    cls_names = [f"class_{c}" for c in range(1, n_cls)]
    # Use meaningful names for known datasets
    if args.dataset in ("busi", "isic", "synthetic_binary"):
        cls_names = ["lesion"]
    elif args.dataset in ("synapse", "synthetic_ct"):
        cls_names = ["aorta", "gallbladder", "spleen", "left_kidney",
                     "right_kidney", "liver", "stomach", "pancreas"]
    elif args.dataset in ("brats", "synthetic_mri"):
        cls_names = ["NCR", "ED", "ET"]

    print("\n  Per-class DSC:")
    print(f"  {'Class':<15} {'DSC':>8}")
    print("  " + "-" * 25)
    for i, name in enumerate(cls_names):
        c  = i + 1
        p  = (preds_all == c)
        t  = (trues_all == c)
        d  = (2*(p&t).sum() + 1e-5) / (p.sum() + t.sum() + 1e-5)
        per_class_dsc[name] = float(d)
        print(f"  {name:<15} {d*100:>7.2f}%")

    # Save results CSV
    result_path = os.path.join(args.out_dir, "results.csv")
    with open(result_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["mean_dsc",  f"{mean_dsc:.4f}"])
        w.writerow(["mean_iou",  f"{mean_iou:.4f}"])
        w.writerow(["precision", f"{prec:.4f}"])
        w.writerow(["recall",    f"{rec:.4f}"])
        w.writerow(["f1_score",  f"{f1:.4f}"])
        for name, val in per_class_dsc.items():
            w.writerow([f"dsc_{name}", f"{val:.4f}"])
    print(f"\n  Results saved -> {result_path}")

    # Plots
    plot_per_class(per_class_dsc, args.out_dir)
    plot_predictions(model, test_loader, n_cls, device, args.out_dir, n=4)
    plot_comparison({"mean_dsc": mean_dsc}, args.out_dir)
    plot_confusion(preds_all, trues_all, n_cls, args.out_dir)

    print(f"\nAll outputs saved to: {args.out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
