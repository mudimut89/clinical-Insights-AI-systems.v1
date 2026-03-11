from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from .config import TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class ImageFolderLikeDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def find_image_roots(extracted_dir: Path) -> List[Path]:
    candidates = []
    for p in extracted_dir.rglob("*"):
        if p.is_dir():
            # If a directory contains image files and subdirs, it may be root.
            img_files = list(p.glob("*.jpg")) + list(p.glob("*.jpeg")) + list(p.glob("*.png"))
            subdirs = [d for d in p.iterdir() if d.is_dir()]
            if img_files and not subdirs:
                continue
            def _subdir_has_images(sd: Path) -> bool:
                return (
                    any(sd.glob("*.jpg"))
                    or any(sd.glob("*.jpeg"))
                    or any(sd.glob("*.png"))
                )

            if subdirs and any(_subdir_has_images(sd) for sd in subdirs):
                candidates.append(p)
    # Prefer shallowest candidates
    candidates = sorted(set(candidates), key=lambda x: len(x.parts))
    return candidates


def build_samples(root_dir: Path) -> Tuple[List[Tuple[Path, int]], Dict[str, int]]:
    class_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {root_dir}")

    class_names = sorted([d.name for d in class_dirs])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    samples: List[Tuple[Path, int]] = []
    exts = {".jpg", ".jpeg", ".png"}

    for cls in class_dirs:
        label = class_to_idx[cls.name]
        for img_path in cls.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in exts:
                samples.append((img_path, label))

    if not samples:
        raise ValueError(f"No images found under {root_dir}")

    return samples, class_to_idx


def split_samples(samples: List[Tuple[Path, int]], seed: int) -> Tuple[List, List, List]:
    rng = random.Random(seed)
    rng.shuffle(samples)
    n = len(samples)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train = samples[:n_train]
    val = samples[n_train : n_train + n_val]
    test = samples[n_train + n_val :]
    return train, val, test


def create_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_pred = []
    y_proba = []

    softmax = nn.Softmax(dim=1)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = softmax(logits)
        pred = probs.argmax(dim=1)

        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        y_proba.append(probs.detach().cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_proba)


def save_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = TrainConfig(project_root=Path(args.project_root))
    if args.epochs is not None:
        cfg = TrainConfig(**{**asdict(cfg), "epochs": args.epochs})

    set_seed(cfg.seed)

    raw_dir = cfg.raw_dir
    extracted_dir = raw_dir / "extracted"
    if not extracted_dir.exists():
        raise FileNotFoundError(
            f"Expected extracted dataset at {extracted_dir}. Run: python -m ml.download_dataset"
        )

    roots = find_image_roots(extracted_dir)
    if not roots:
        raise ValueError(f"Could not find image root with class subfolders under: {extracted_dir}")

    # Choose the first candidate root
    data_root = roots[0]
    samples, class_to_idx = build_samples(data_root)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_s, val_s, test_s = split_samples(samples, cfg.seed)

    train_tfms = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = ImageFolderLikeDataset(train_s, transform=train_tfms)
    val_ds = ImageFolderLikeDataset(val_s, transform=eval_tfms)
    test_ds = ImageFolderLikeDataset(test_s, transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(num_classes=len(class_to_idx)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_val_acc = -1.0
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.artifacts_dir / "model_best.pt"

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        losses = []
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)))

        y_true, y_pred, _ = predict(model, val_loader, device)
        val_acc = accuracy_score(y_true, y_pred)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "image_size": cfg.image_size,
                },
                ckpt_path,
            )

    # Load best model
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state_dict"])

    y_true, y_pred, y_proba = predict(model, test_loader, device)

    labels = [idx_to_class[i] for i in range(len(idx_to_class))]

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, labels, cfg.artifacts_dir / "confusion_matrix.png")

    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": report,
    }

    if len(labels) == 2:
        # Pick positive class as index 1
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except Exception:
            metrics["roc_auc"] = None

    (cfg.artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save a small sample of prediction confidences
    confidences = []
    for i in range(min(50, len(y_true))):
        confidences.append(
            {
                "true": int(y_true[i]),
                "pred": int(y_pred[i]),
                "confidence": float(np.max(y_proba[i])),
                "probs": [float(p) for p in y_proba[i]],
            }
        )
    (cfg.artifacts_dir / "confidence_samples.json").write_text(json.dumps(confidences, indent=2))

    # Export TorchScript for backend inference
    model.eval()
    example = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device)
    traced = torch.jit.trace(model, example)
    traced.save(str(cfg.artifacts_dir / "model_ts.pt"))

    (cfg.artifacts_dir / "classes.json").write_text(json.dumps({"class_to_idx": class_to_idx}, indent=2))

    print(f"Saved artifacts to: {cfg.artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
