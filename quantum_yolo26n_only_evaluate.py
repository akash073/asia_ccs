"""Evaluation ONLY: saved hybrid YOLO26n-cls + 4-layer PennyLane head.

Matches checkpoints produced by the user's hybrid YOLO26n training script.
No training; no CNN/DNN-only models; no hardware quantum computer.

Run:
    python yolo26n_quantum_only_evaluate.py
    python yolo26n_quantum_only_evaluate.py --datasets CIFAR100 TinyImageNet200
    python yolo26n_quantum_only_evaluate.py --test-samples 100
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score)
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from ultralytics import YOLO

N_Q_LAYERS = 4
DEVICE = torch.device('cpu')  # PennyLane statevector execution on CPU
CONFIGS = {
    'MNIST': (datasets.MNIST, 'mnist', 10),
    'FashionMNIST': (datasets.FashionMNIST, 'fashionmnist', 10),
    'KMNIST': (datasets.KMNIST, 'kmnist', 10),
    'CIFAR100': (datasets.CIFAR100, 'cifar100', 100),
    'TinyImageNet200': (None, 'tinyimagenet200', 200),
}
KEY_TO_NAME = {v[1]: k for k, v in CONFIGS.items()}
PATTERN = re.compile(
    r'^hybrid_yolo26n_(mnist|fashionmnist|kmnist|cifar100|tinyimagenet200)_(\d+)q_(\d+)layers_(\d+)px_train(\d+)_v3\.pth$', re.I)


class TinyImageNetDataset(Dataset):
    """Same train-class indexing and val annotations as original hybrid YOLO code."""
    def __init__(self, root: Path, transform):
        wnids_file = root / 'wnids.txt'
        if not wnids_file.is_file():
            raise FileNotFoundError(wnids_file)
        wnids = sorted(s.strip() for s in wnids_file.read_text().splitlines() if s.strip())
        if len(wnids) != 200:
            raise ValueError('Tiny ImageNet requires 200 class IDs in wnids.txt')
        class_ids = {wnid: i for i, wnid in enumerate(wnids)}
        ann = root / 'val' / 'val_annotations.txt'
        if not ann.exists():
            raise FileNotFoundError(ann)
        self.samples = []
        for line in ann.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] in class_ids:
                image_path = root / 'val' / 'images' / parts[0]
                if image_path.is_file():
                    self.samples.append((image_path, class_ids[parts[1]]))
        if not self.samples:
            raise ValueError('No labeled validation images found for Tiny ImageNet')
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        with Image.open(filename) as image:
            return self.transform(image.convert('RGB')), label


def load_test_dataset(name, image_size, data_root, tiny_root):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.ToTensor(),
    ])
    if name == 'TinyImageNet200':
        return TinyImageNetDataset(tiny_root, transform)
    dataset_class = CONFIGS[name][0]
    return dataset_class(root=str(data_root), train=False, download=True, transform=transform)


def make_qnode(n_qubits):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev, interface='torch', diff_method='best')
    def circuit(x, weights):
        for wire in range(n_qubits):
            qml.RY(x[wire], wires=wire)
            qml.RZ(x[wire], wires=wire)
        for layer in range(N_Q_LAYERS):
            for wire in range(n_qubits):
                qml.Rot(*weights[layer, wire], wires=wire)
            if n_qubits > 1:
                for wire in range(n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]
    return circuit


class QuantumYOLOHead(nn.Module):
    """Exact module structure used in uploaded YOLO hybrid training code."""
    def __init__(self, in_features, num_classes, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.feature_to_qubits = nn.Linear(in_features, n_qubits)
        self.q_weights = nn.Parameter(torch.randn(N_Q_LAYERS, n_qubits, 3) * .01)
        self.classifier = nn.Linear(n_qubits, num_classes)
        self.qnode = make_qnode(n_qubits)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        angles = torch.tanh(self.feature_to_qubits(x)) * torch.pi
        rows = []
        for one in angles:
            result = self.qnode(one, self.q_weights)
            features = result.reshape(-1) if isinstance(result, torch.Tensor) else torch.stack(list(result)).reshape(-1)
            rows.append(features.to(dtype=self.classifier.weight.dtype))
        return self.classifier(torch.stack(rows, dim=0))


def get_last_linear(model):
    matches = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]
    if not matches:
        raise RuntimeError('YOLO26n classification architecture does not contain nn.Linear')
    name, old = matches[-1]
    parent = model
    for part in name.split('.')[:-1]:
        parent = parent[int(part)] if part.isdecimal() else getattr(parent, part)
    return parent, name.split('.')[-1], old


def build_model(n_classes, n_qubits):
    # Intentionally use scratch architecture: matches original training source,
    # which called YOLO('yolo26n-cls.yaml'), NOT pretrained .pt.
    model = YOLO('yolo26n-cls.yaml').model
    parent, key, old = get_last_linear(model)
    head = QuantumYOLOHead(old.in_features, n_classes, n_qubits)
    if key.isdecimal():
        parent[int(key)] = head
    else:
        setattr(parent, key, head)
    return model.to(DEVICE).eval()


def get_logits(output, n_classes):
    if isinstance(output, torch.Tensor) and output.ndim == 2 and output.shape[1] == n_classes:
        return output
    if isinstance(output, (tuple, list)):
        candidates = [x for x in output if isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[1] == n_classes]
        if candidates:
            return candidates[-1]  # original hybrid YOLO helper uses last tensor
    if isinstance(output, dict):
        for key in ('logits', 'preds', 'output'):
            if key in output:
                return get_logits(output[key], n_classes)
    raise RuntimeError(f'Unexpected YOLO output type/shape: {type(output)}')


def checkpoint_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


def write_csv(df, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    df.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def metrics(y_true, y_pred, num_classes):
    labels = list(range(num_classes))
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, default=Path('quantum_checkpoints'))
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--tiny-root', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=Path('evaluation_results') / 'yolo26n_quantum')
    parser.add_argument('--datasets', nargs='*', choices=list(CONFIGS), default=None)
    parser.add_argument('--test-samples', type=int, default=0, help='0 = entire held-out test split')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--min-epochs', type=int, default=10, help='Minimum saved epoch; use 1 to include partial models')
    parser.add_argument('--force', action='store_true', help='Recompute already saved predictions')
    args = parser.parse_args()
    if args.test_samples < 0 or args.batch_size < 1 or args.min_epochs < 1:
        parser.error('Invalid evaluation limits')
    tiny_root = args.tiny_root or (args.data_root / 'tiny-imagenet-200')
    paths = sorted(args.checkpoints.rglob('hybrid_yolo26n_*.pth'))
    if not paths:
        raise FileNotFoundError(f'No hybrid YOLO .pth checkpoints in {args.checkpoints.resolve()}')

    summary_path = args.output / 'yolo26n_quantum_accuracy_summary.csv'
    failures_path = args.output / 'skipped_checkpoints.csv'
    results = pd.read_csv(summary_path).to_dict('records') if summary_path.exists() else []
    failures = []
    cache = {}
    for checkpoint_path in paths:
        match = PATTERN.fullmatch(checkpoint_path.name)
        if match is None:
            continue
        dataset_key, qubits, layers, img_size, training_count = match.groups()
        name = KEY_TO_NAME[dataset_key.lower()]
        if args.datasets and name not in args.datasets:
            continue
        qubits, layers, img_size, training_count = map(int, (qubits, layers, img_size, training_count))
        print(f'\n{checkpoint_path.name}')
        try:
            if layers != N_Q_LAYERS:
                raise ValueError(f'Expected {N_Q_LAYERS} quantum layers, found {layers}')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
                raise ValueError('Checkpoint has no state_dict')
            n_classes = CONFIGS[name][2]
            for key, value in [('n_qubits', qubits), ('n_classes', n_classes), ('train_count', training_count)]:
                if int(checkpoint.get(key, -1)) != value:
                    raise ValueError(f'Checkpoint metadata mismatch: {key}')
            epoch = int(checkpoint.get('epoch', -1))
            if epoch < args.min_epochs:
                raise ValueError(f'Checkpoint epoch {epoch} < required {args.min_epochs}')

            cache_key = (name, img_size)
            if cache_key not in cache:
                cache[cache_key] = load_test_dataset(name, img_size, args.data_root, tiny_root)
            ds = cache[cache_key]
            count = len(ds) if args.test_samples == 0 else min(len(ds), args.test_samples)
            digest = checkpoint_digest(checkpoint_path)
            pred_path = args.output / 'predictions' / f'{checkpoint_path.stem}_{digest}_test{count}.csv'
            if pred_path.exists() and not args.force:
                frame = pd.read_csv(pred_path)
                if frame['sample_index'].duplicated().any():
                    raise ValueError('Duplicated sample_index in saved predictions')
                done = set(frame['sample_index'].astype(int).tolist())
                if not done.issubset(set(range(count))):
                    raise ValueError('Saved prediction indices outside expected test set')
                rows = frame.to_dict('records')
            else:
                done, rows = set(), []
            pending = [i for i in range(count) if i not in done]
            if pending:
                model = build_model(n_classes, qubits)
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                model.eval()
                loader = DataLoader(Subset(ds, pending), batch_size=args.batch_size, shuffle=False, num_workers=0)
                offset = 0
                with torch.no_grad():
                    for images, targets in tqdm(loader, desc=f'{name}/{qubits}q'):
                        logits = get_logits(model(images.to(DEVICE)), n_classes)
                        predictions = logits.argmax(1).cpu().tolist()
                        for j, (actual, predicted) in enumerate(zip(targets.tolist(), predictions)):
                            rows.append({'sample_index': pending[offset+j], 'true_label': int(actual),
                                         'prediction': int(predicted), 'correct': int(actual == predicted)})
                        offset += len(targets)
                        if offset % 100 < len(targets):
                            write_csv(pd.DataFrame(rows).sort_values('sample_index'), pred_path)
                del model
                gc.collect()
            frame = pd.DataFrame(rows).sort_values('sample_index')
            if len(frame) != count or frame.sample_index.astype(int).tolist() != list(range(count)):
                raise RuntimeError('Missing, duplicated or out-of-order test samples')
            write_csv(frame, pred_path)
            scores = metrics(frame.true_label.astype(int), frame.prediction.astype(int), n_classes)
            result = dict(dataset=name, model_type='HybridYOLO26n-QuantumHead',
                          quantum_simulation=True, n_qubits=qubits, quantum_layers=layers,
                          image_size=img_size, epochs_saved=epoch, training_samples=training_count,
                          test_samples=count, total_classes=n_classes,
                          checkpoint_path=str(checkpoint_path.resolve()), checkpoint_hash=digest,
                          prediction_csv=str(pred_path.resolve()), **scores)
            results = [r for r in results if r.get('checkpoint_path') != result['checkpoint_path'] or
                       int(r.get('test_samples', -1)) != count]
            results.append(result)
            write_csv(pd.DataFrame(results).sort_values(['dataset', 'n_qubits'], ascending=[True, False]), summary_path)
            print(f"  accuracy={scores['accuracy']:.4f}, balanced_accuracy={scores['balanced_accuracy']:.4f}, "
                  f"macro_F1={scores['f1_macro']:.4f} ({count} images)")
        except Exception as exc:
            print(f'  SKIPPED / ERROR: {exc}')
            failures.append({'checkpoint_path': str(checkpoint_path), 'reason': str(exc)})
            write_csv(pd.DataFrame(failures), failures_path)
    print(f'\nSaved summary: {summary_path} ({len(results)} rows)')
    if failures:
        print(f'Skipped/failed checkpoints: {failures_path}')


if __name__ == '__main__':
    main()
