#!/usr/bin/env python3
"""Pretrained YOLO26n-CLS: 10-qubit PennyLane vs two matched classical controls.

Features:
- Models: quantum, classical_bottleneck, classical_standard.
- Pretrained ImageNet YOLO26n-CLS backbone for every configuration.
- Stage 1: train new head with a frozen backbone.
- Stage 2: fine-tune backbone at a lower learning rate.
- Stratified reproducible validation split; official held-out test/val only once.
- Complete Tiny ImageNet audit (200 classes, 100000 training, 10000 validation).
- Resume from last-epoch checkpoints; save best validation weights separately.
- Metrics, per-sample predictions, and confusion matrices; NO energy telemetry.

Examples:
  python train_improved_hybrid_yolo26n_10q.py --datasets CIFAR100
  python train_improved_hybrid_yolo26n_10q.py --datasets CIFAR100 TinyImageNet200 --models quantum classical_bottleneck classical_standard
  python train_improved_hybrid_yolo26n_10q.py --datasets CIFAR100 --train-samples 1000 --test-samples 100 --head-epochs 1 --finetune-epochs 1
  python train_improved_hybrid_yolo26n_10q.py --datasets CIFAR100 --eval-only

The quantum circuit uses classical CPU simulation; no quantum processor is used.
Do not compare telemetry collected here against earlier randomly initialized architectures
without distinguishing training conditions and hardware.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

try:
    import pennylane as qml
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}. Install with: "
        "pip install ultralytics pennylane torch torchvision scikit-learn pandas pillow tqdm"
    ) from exc

DATASET_SPEC = {
    "CIFAR10": (10, "cifar10"),
    "MNIST": (10, "mnist"),
    "FashionMNIST": (10, "fashionmnist"),
    "KMNIST": (10, "kmnist"),
    
    #"CIFAR100": (100, "cifar100"),
    #"TinyImageNet200": (200, "tinyimagenet200"),
}
MODEL_KINDS = ("quantum", "classical_bottleneck", "classical_standard")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_SPEC), default=list(DATASET_SPEC))
    parser.add_argument("--models", nargs="+", choices=MODEL_KINDS,
                        default=["quantum", "classical_bottleneck", "classical_standard"])
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--tiny-root", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("quantum_checkpoints/pretrained_5q"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results/pretrained_5q"))
    parser.add_argument("--pretrained-weights", default="yolo26n-cls.pt")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Pretrained model's standard classification resolution; same for all models")
    parser.add_argument("--qubits", type=int, default=5)
    parser.add_argument("--quantum-layers", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-finetune-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--train-samples", type=int, default=0,
                        help="0 uses complete official training set before validation split")
    parser.add_argument("--test-samples", type=int, default=0,
                        help="0 evaluates complete official test set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate saved best checkpoint; NEVER train")
    parser.add_argument("--no-augmentation", action="store_true")
    # Jupyter adds -f kernel.json. Do not ignore ordinary CLI errors.
    argv = sys.argv[1:]
    if "ipykernel" in sys.modules:
        clean = []
        i = 0
        while i < len(argv):
            value = argv[i]
            if value in ("-f", "--f"):
                i += 2
            elif re.match(r"^(?:-f|--f)=.*\.json$", value):
                i += 1
            else:
                clean.append(value)
                i += 1
        argv = clean
    args = parser.parse_args(argv)
    if args.qubits < 1 or args.quantum_layers < 1:
        parser.error("--qubits and --quantum-layers must be positive")
    if args.batch_size < 1 or args.workers < 0 or args.image_size < 64:
        parser.error("Invalid batch-size/workers/image-size")
    if args.head_epochs < 0 or args.finetune_epochs < 0 or (args.head_epochs + args.finetune_epochs == 0 and not args.eval_only):
        parser.error("At least one stage must have a positive number of epochs")
    if not 0 < args.val_fraction < 1:
        parser.error("A positive validation fraction is required to select best checkpoints without test leakage")
    if args.train_samples < 0 or args.test_samples < 0:
        parser.error("Sample limits must be nonnegative")
    args.data_root = args.data_root.expanduser().resolve()
    args.tiny_root = (args.tiny_root or args.data_root / "tiny-imagenet-200").expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(args):
    if args.device == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected = args.device
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if "quantum" in args.models and selected != "cpu":
        raise RuntimeError("For fair CPU-based simulated-quantum comparison, use --device cpu")
    return torch.device(selected)


class TinyImageNet(Dataset):
    def __init__(self, root: Path, split: str, transform=None):
        self.transform = transform
        self.samples = []
        self.root = root
        ids_path = root / "wnids.txt"
        if not ids_path.is_file():
            raise FileNotFoundError(f"Tiny ImageNet class index missing: {ids_path}")
        classes = sorted(x.strip() for x in ids_path.read_text(encoding="utf-8").splitlines() if x.strip())
        if len(classes) != 200 or len(set(classes)) != 200:
            raise ValueError("Tiny ImageNet wnids.txt must contain 200 distinct classes")
        mapping = {c: i for i, c in enumerate(classes)}
        self.targets = []
        if split == "train":
            for wnid in classes:
                folder = root / "train" / wnid / "images"
                paths = sorted(p for p in folder.glob("*") if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
                if len(paths) != 500:
                    raise RuntimeError(f"Tiny ImageNet training class {wnid}: expected 500 images, got {len(paths)} in {folder}")
                self.samples.extend((p, mapping[wnid]) for p in paths)
        elif split == "val":
            annotations = root / "val" / "val_annotations.txt"
            if not annotations.is_file():
                raise FileNotFoundError(f"Tiny ImageNet validation annotations missing: {annotations}")
            lines = annotations.read_text(encoding="utf-8").splitlines()
            if len(lines) != 10_000:
                raise RuntimeError(f"Tiny ImageNet validation expected 10000 annotations, found {len(lines)}")
            missing = []
            for line in lines:
                fields = line.split()
                if len(fields) < 2 or fields[1] not in mapping:
                    raise RuntimeError(f"Invalid Tiny ImageNet annotation: {line}")
                path = root / "val" / "images" / fields[0]
                if not path.is_file():
                    missing.append(str(path))
                self.samples.append((path, mapping[fields[1]]))
            counts = Counter(label for _, label in self.samples)
            if len(counts) != 200 or any(counts.get(i, 0) != 50 for i in range(200)):
                raise RuntimeError("Tiny ImageNet validation must have exactly 50 images in each of 200 classes")
            if len(set(str(p) for p, _ in self.samples)) != 10_000:
                raise RuntimeError("Tiny ImageNet validation has duplicate annotation filenames")
            if missing:
                raise RuntimeError(f"Tiny ImageNet validation is incomplete ({len(missing)} files missing). "
                                   f"First missing: {missing[:5]}")
            # Fixed canonical order ensures sample IDs consistently map to images.
            self.samples.sort(key=lambda x: x[0].name)
        else:
            raise ValueError(split)
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as img:
            image = img.convert("RGB")
            return self.transform(image) if self.transform else image, label


class ApplyTransform(Dataset):
    """Same underlying official examples with train vs eval transforms."""
    def __init__(self, base, indices, transform):
        self.base = base
        self.indices = [int(i) for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, target = self.base[self.indices[i]]
        return self.transform(image), int(target)


def build_transforms(size, augmentation, dataset_name):
    # Pretrained YOLO cls weights were optimized on ImageNet-like image colors.
    # Inputs are RGB 0..1; same transforms for quantum and classical controls.
    eval_transform = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
    ])
    # Horizontal flipping would change the meaning of some handwritten digits.
    augmentations = []
    if augmentation:
        if dataset_name in ("MNIST", "FashionMNIST", "KMNIST"):
            augmentations = [transforms.RandomRotation(degrees=8)]
        else:
            augmentations = [transforms.RandomHorizontalFlip(p=0.5),
                             transforms.RandomRotation(degrees=8)]
    train_transform = transforms.Compose([
        transforms.Resize((size, size)), *augmentations, transforms.ToTensor(),
    ])
    return train_transform, eval_transform


def dataset_base(name, is_train, args):
    if name == "TinyImageNet200":
        return TinyImageNet(args.tiny_root, "train" if is_train else "val", transform=None)
    ds_cls = {"MNIST": datasets.MNIST, "FashionMNIST": datasets.FashionMNIST,
              "KMNIST": datasets.KMNIST, "CIFAR10": datasets.CIFAR10,
              "CIFAR100": datasets.CIFAR100}[name]
    return ds_cls(root=str(args.data_root), train=is_train, download=True, transform=None)


def label_array(ds):
    if hasattr(ds, "targets"):
        return np.asarray(ds.targets, dtype=np.int64)
    return np.asarray([int(ds[i][1]) for i in range(len(ds))], dtype=np.int64)


def dataset_loaders(name, args):
    train_base = dataset_base(name, True, args)
    test_base = dataset_base(name, False, args)
    labels = label_array(train_base)
    indices = np.arange(len(labels))
    # Sample at most train_samples, with reproducible stratification by class.
    rng = np.random.default_rng(args.seed)
    if args.train_samples and args.train_samples < len(indices):
        selected = []
        # approximately balanced class sampling if using a restricted subset
        n_class = DATASET_SPEC[name][0]
        base_quota = args.train_samples // n_class
        extra = args.train_samples % n_class
        for c in range(n_class):
            ids = indices[labels == c]
            chosen = rng.choice(ids, size=min(len(ids), base_quota + int(c < extra)), replace=False)
            selected.extend(int(x) for x in chosen)
        if len(selected) != args.train_samples:
            raise ValueError("Subset sampling failed to cover all requested examples")
        indices = np.asarray(selected, dtype=np.int64)
    # Avoid split failure if a tiny debug sample contains fewer than two examples per class.
    from sklearn.model_selection import train_test_split
    class_counts = Counter(labels[indices].tolist())
    if min(class_counts.values()) < 2:
        raise ValueError("Need at least two training examples per class for stratified validation; increase --train-samples")
    train_idx, val_idx = train_test_split(indices, test_size=args.val_fraction,
                                           stratify=labels[indices], random_state=args.seed)
    train_t, eval_t = build_transforms(args.image_size, not args.no_augmentation, name)
    # ImageNet pretrained conv backbone expects 3-channel RGB.
    to_rgb = transforms.Lambda(lambda image: image.convert("RGB"))
    train_t.transforms.insert(0, to_rgb)
    eval_t.transforms.insert(0, to_rgb)
    train_ds = ApplyTransform(train_base, train_idx, train_t)
    val_ds = ApplyTransform(train_base, val_idx, eval_t)
    test_indices = np.arange(len(test_base))
    if args.test_samples:
        test_indices = test_indices[:args.test_samples]
    test_ds = ApplyTransform(test_base, test_indices, eval_t)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=False)
    train_dl = DataLoader(train_ds, shuffle=True, **kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **kwargs)
    test_dl = DataLoader(test_ds, shuffle=False, **kwargs)
    print(f"{name}: train={len(train_ds)}, validation={len(val_ds)}, test={len(test_ds)} ",
          f"(official test={len(test_base)}); all {DATASET_SPEC[name][0]} classes")
    return train_dl, val_dl, test_dl, len(train_base), len(test_base)


def circuit_factory(n_qubits, n_layers):
    device = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(device, interface="torch", diff_method="best")
    def circuit(x, weights):
        # Angle encoding at every layer = data re-uploading.
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(x[q], wires=q)
                qml.RZ(x[q], wires=q)
                qml.Rot(weights[layer, q, 0], weights[layer, q, 1],
                        weights[layer, q, 2], wires=q)
            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
    return circuit


class QuantumHead(nn.Module):
    def __init__(self, in_features, n_classes, n_qubits, n_layers):
        super().__init__()
        self.feature_to_qubits = nn.Linear(in_features, n_qubits)
        self.q_weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))
        self.classifier = nn.Linear(n_qubits, n_classes)
        self.qnode = circuit_factory(n_qubits, n_layers)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        angles = torch.tanh(self.feature_to_qubits(x)) * torch.pi
        outputs = []
        for one in angles:
            values = self.qnode(one, self.q_weights)
            if isinstance(values, torch.Tensor):
                q_out = values.reshape(-1)
            else:
                q_out = torch.stack(list(values)).reshape(-1)
            outputs.append(q_out.to(dtype=self.classifier.weight.dtype))
        return self.classifier(torch.stack(outputs, dim=0))


class BottleneckControl(nn.Module):
    def __init__(self, in_features, n_classes, width):
        super().__init__()
        self.feature_to_qubits = nn.Linear(in_features, width)
        self.classifier = nn.Linear(width, n_classes)

    def forward(self, x):
        return self.classifier(torch.tanh(self.feature_to_qubits(x)))


def build_model(kind, n_classes, args):
    # All three contenders start from the identical pretrained architecture.
    yolo = YOLO(args.pretrained_weights)
    model = yolo.model
    candidate = []
    for path, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            candidate.append((path, mod))
    if not candidate:
        raise RuntimeError("Pretrained YOLO classification model contains no final nn.Linear")
    linear_path, original_linear = candidate[-1]
    if not linear_path.endswith(".linear"):
        raise RuntimeError(f"Expected Ultralytics Classify.linear, found {linear_path}. "
                           "Check installed ultralytics version.")
    if kind == "quantum":
        head = QuantumHead(original_linear.in_features, n_classes, args.qubits, args.quantum_layers)
    elif kind == "classical_bottleneck":
        head = BottleneckControl(original_linear.in_features, n_classes, args.qubits)
    elif kind == "classical_standard":
        head = nn.Linear(original_linear.in_features, n_classes)
    else:
        raise ValueError(kind)
    parent = model
    keys = linear_path.split(".")
    for item in keys[:-1]:
        parent = parent[int(item)] if item.isdigit() else getattr(parent, item)
    setattr(parent, keys[-1], head)
    return model, head


def logits_from_forward(output, n_classes):
    # Ultralytics Classify: train -> logits; eval -> (probabilities, logits).
    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, (tuple, list)) and len(output) >= 2 and isinstance(output[1], torch.Tensor):
        logits = output[1]
    elif isinstance(output, dict) and isinstance(output.get("logits"), torch.Tensor):
        logits = output["logits"]
    else:
        raise RuntimeError(f"Unexpected YOLO classification output: {type(output)}")
    if logits.ndim != 2 or logits.shape[1] != n_classes:
        raise RuntimeError(f"Invalid classification output shape {tuple(logits.shape)} expected (?, {n_classes})")
    return logits


def set_training_stage(model, head, stage):
    if stage == "head":
        # Keep frozen backbone BatchNorm running statistics unchanged.
        for p in model.parameters():
            p.requires_grad_(False)
        for p in head.parameters():
            p.requires_grad_(True)
        model.eval()
        head.train()
    elif stage == "finetune":
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
    else:
        raise ValueError(stage)


def optimizer_for_stage(model, head, stage, args):
    if stage == "head":
        return torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr},
        {"params": list(head.parameters()), "lr": args.head_finetune_lr},
    ], weight_decay=args.weight_decay)


def train_epoch(model, head, loader, optimizer, stage, n_classes, device):
    set_training_stage(model, head, stage)
    total_loss = correct = count = 0
    for images, targets in tqdm(loader, desc=f"Train {stage}", leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = logits_from_forward(model(images), n_classes)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        batch = len(targets)
        count += batch
        total_loss += float(loss.detach().item()) * batch
        correct += int((logits.argmax(1) == targets).sum().item())
    return total_loss / max(1, count), correct / max(1, count)


@torch.no_grad()
def validation_epoch(model, loader, n_classes, device):
    model.eval()
    loss_sum = correct = count = 0
    for images, targets in tqdm(loader, desc="Validation", leave=False):
        images, targets = images.to(device), targets.to(device)
        logits = logits_from_forward(model(images), n_classes)
        batch = len(targets)
        loss_sum += float(F.cross_entropy(logits, targets).item()) * batch
        correct += int((logits.argmax(1) == targets).sum().item())
        count += batch
    return loss_sum / max(1, count), correct / max(1, count)


def checkpoint_paths(name, kind, args):
    prefix = (f"{DATASET_SPEC[name][1]}_{kind}_{args.qubits}q_"
              f"{args.quantum_layers}layers_{args.image_size}px_seed{args.seed}_"
              f"pretrained_v1")
    return (args.checkpoint_dir / f"{prefix}_last.pth",
            args.checkpoint_dir / f"{prefix}_best.pth")


def config_digest(args, name, kind):
    payload = {k: getattr(args, k) for k in (
        "qubits", "quantum_layers", "image_size", "seed", "train_samples", "val_fraction",
        "head_epochs", "finetune_epochs", "head_lr", "backbone_lr",
        "head_finetune_lr", "batch_size", "weight_decay", "pretrained_weights",
        "no_augmentation")}
    payload.update({"dataset": name, "kind": kind})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def save_checkpoint(path, model, optimizer, metadata):
    temp = path.with_suffix(".tmp")
    torch.save({**metadata, "state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict() if optimizer is not None else None}, temp)
    os.replace(temp, path)


def load_checkpoint(path, device, digest):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get("config_digest") != digest:
        raise RuntimeError(f"Checkpoint configuration mismatch: {path}. "
                           "Use a separate checkpoint directory or consistent flags.")
    return ckpt


def fit_model(name, kind, args, train_dl, val_dl, device):
    n_classes = DATASET_SPEC[name][0]
    last_path, best_path = checkpoint_paths(name, kind, args)
    digest = config_digest(args, name, kind)
    model, head = build_model(kind, n_classes, args)
    model.to(device)
    stages = [("head", args.head_epochs), ("finetune", args.finetune_epochs)]
    total_steps = sum(n for _, n in stages)
    start = 0
    best_val = -1.0
    resume = None
    if last_path.is_file():
        resume = load_checkpoint(last_path, device, digest)
        model.load_state_dict(resume["state_dict"], strict=True)
        start = int(resume["completed_epochs"])
        best_val = float(resume.get("best_val_accuracy", -1.0))
        if start >= total_steps:
            print(f"Already trained {name}/{kind}: {start}/{total_steps} epochs; loading best")
    elif args.eval_only and not best_path.is_file():
        raise FileNotFoundError(f"No saved checkpoint for evaluation: {best_path}")
    if not args.eval_only and start < total_steps:
        for global_epoch in range(start, total_steps):
            stage = "head" if global_epoch < args.head_epochs else "finetune"
            stage_index = global_epoch if stage == "head" else global_epoch - args.head_epochs
            # Initializing an optimizer per stage is deliberate; restore it on resume mid-stage.
            if global_epoch == start or stage_index == 0:
                optimizer = optimizer_for_stage(model, head, stage, args)
                if resume is not None and global_epoch == start and stage_index > 0:
                    if resume.get("optimizer_state") is not None:
                        optimizer.load_state_dict(resume["optimizer_state"])
            train_loss, train_acc = train_epoch(model, head, train_dl, optimizer, stage, n_classes, device)
            val_loss, val_acc = validation_epoch(model, val_dl, n_classes, device)
            if val_acc > best_val:
                best_val = val_acc
                save_checkpoint(best_path, model, None, {
                    "dataset": name, "kind": kind, "qubits": args.qubits,
                    "n_classes": n_classes, "config_digest": digest,
                    "selected_epoch": global_epoch + 1, "best_val_accuracy": best_val,
                    "pretrained_weights": args.pretrained_weights,
                    "train_count": len(train_dl.dataset), "val_count": len(val_dl.dataset),
                })
            save_checkpoint(last_path, model, optimizer, {
                "dataset": name, "kind": kind, "qubits": args.qubits,
                "n_classes": n_classes, "config_digest": digest,
                "completed_epochs": global_epoch + 1, "total_epochs": total_steps,
                "stage": stage, "best_val_accuracy": best_val,
                "pretrained_weights": args.pretrained_weights,
                "train_count": len(train_dl.dataset), "val_count": len(val_dl.dataset),
            })
            print(f"{name}/{kind} epoch={global_epoch + 1}/{total_steps} stage={stage}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, best_val_acc={best_val:.4f}")
    if not best_path.is_file():
        raise FileNotFoundError(f"Best checkpoint missing: {best_path}")
    best = load_checkpoint(best_path, device, digest)
    model.load_state_dict(best["state_dict"], strict=True)
    model.eval()
    return model, best_path, best


@torch.no_grad()
def evaluate_test(name, kind, model, best_path, best, test_dl, official_test_len, args, device):
    model.eval()
    n_classes = DATASET_SPEC[name][0]
    true, pred, records = [], [], []
    index = 0
    for images, targets in tqdm(test_dl, desc=f"TEST {name}/{kind}", leave=False):
        images = images.to(device)
        logits = logits_from_forward(model(images), n_classes)
        prob = logits.float().softmax(dim=1)
        predicted = logits.argmax(dim=1).cpu().tolist()
        confidence = prob.max(dim=1).values.cpu().tolist()
        for gt, p, conf in zip(targets.tolist(), predicted, confidence):
            true.append(int(gt))
            pred.append(int(p))
            records.append({"sample_index": index, "true_label": int(gt),
                            "prediction": int(p), "correct": int(gt == p),
                            "confidence": float(conf)})
            index += 1
    output = args.output_dir
    basename = best_path.stem.removesuffix("_best")
    pred_path = output / "predictions" / f"{basename}_predictions.csv"
    confusion_path = output / "confusion_matrices" / f"{basename}_confusion_matrix.csv"
    pd.DataFrame(records).to_csv(pred_path, index=False)
    pd.DataFrame(confusion_matrix(true, pred, labels=range(n_classes)),
                 index=range(n_classes), columns=range(n_classes)).to_csv(confusion_path, index_label="true\\pred")
    labels = list(range(n_classes))
    metrics = {
        "dataset": name, "model": kind, "n_classes": n_classes,
        "n_qubits_or_bottleneck": args.qubits if kind != "classical_standard" else None,
        "quantum_layers": args.quantum_layers if kind == "quantum" else 0,
        "selected_epoch": best.get("selected_epoch"),
        "best_validation_accuracy": best.get("best_val_accuracy"),
        "train_samples": best.get("train_count"), "validation_samples": best.get("val_count"),
        "test_samples": len(true), "official_test_samples": official_test_len,
        "full_test_set": len(true) == official_test_len,
        "correct_predictions": sum(int(a == b) for a, b in zip(true, pred)),
        "accuracy": accuracy_score(true, pred),
        "balanced_accuracy": balanced_accuracy_score(true, pred),
        "precision_macro": precision_score(true, pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(true, pred, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(true, pred, labels=labels, average="macro", zero_division=0),
        "precision_weighted": precision_score(true, pred, labels=labels, average="weighted", zero_division=0),
        "recall_weighted": recall_score(true, pred, labels=labels, average="weighted", zero_division=0),
        "f1_weighted": f1_score(true, pred, labels=labels, average="weighted", zero_division=0),
        "checkpoint": str(best_path), "predictions_csv": str(pred_path),
        "confusion_matrix_csv": str(confusion_path),
    }
    print(f"RESULT {name}/{kind}: accuracy={metrics['accuracy']:.4f} "
          f"balanced={metrics['balanced_accuracy']:.4f} F1-macro={metrics['f1_macro']:.4f} "
          f"({len(true)} test images)")
    return metrics


def main():
    args = parse_arguments()
    seed_everything(args.seed)
    device = get_device(args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    print("Pretrained Hybrid YOLO26n-CLS / PennyLane statevector simulation")
    print(f"Device={device}; models={args.models}; datasets={args.datasets}")
    print(f"Stages: frozen-head {args.head_epochs} epochs + fine-tune {args.finetune_epochs} epochs")
    print(f"Architecture: {args.qubits}-qubit variational circuit, {args.quantum_layers} layers with data re-uploading")
    print("Note: simulated quantum model on CPU; it may be very slow on full CIFAR-100/Tiny ImageNet")
    results = []
    errors = []
    for name in args.datasets:
        try:
            train_dl, val_dl, test_dl, original_train, original_test = dataset_loaders(name, args)
        except Exception as e:
            print(f"DATASET FAILED {name}: {e}")
            errors.append({"dataset": name, "model": "ALL", "reason": str(e)})
            continue
        for kind in args.models:
            model = None
            try:
                # Re-seed before each initial model for reproducible head initialization.
                seed_everything(args.seed)
                model, path, best = fit_model(name, kind, args, train_dl, val_dl, device)
                result = evaluate_test(name, kind, model, path, best, test_dl, original_test, args, device)
                results.append(result)
                pd.DataFrame(results).to_csv(args.output_dir / "model_accuracy_summary.csv", index=False)
            except Exception as e:
                print(f"FAILED {name}/{kind}: {type(e).__name__}: {e}")
                errors.append({"dataset": name, "model": kind, "reason": f"{type(e).__name__}: {e}"})
                pd.DataFrame(errors).to_csv(args.output_dir / "failures.csv", index=False)
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    if results:
        print(f"\nSaved {len(results)} results: {args.output_dir / 'model_accuracy_summary.csv'}")
    if errors:
        print(f"Some experiments failed: {args.output_dir / 'failures.csv'}")


if __name__ == "__main__":
    main()
