"""Unified CPU inference and telemetry for saved classical and simulated-quantum models.

Save the uploaded quantum training implementation as quantum_dnn_cnn.py so its
model architecture, checkpoint loader and preprocessing are reused exactly.
This script NEVER trains models.
"""
from __future__ import annotations

import hashlib
import os
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psutil
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchvision import datasets, transforms
from tqdm.auto import tqdm

import classical_cnn_dnn as classical
import classical_yolo
import quantum_dnn_cnn as quantum
import quantum_yolo

try:
    from codecarbon import EmissionsTracker
except ImportError:
    EmissionsTracker = None

# ========================== USER SETTINGS ==========================
SAMPLES = 10             # 0 = ALL official test/validation images
GROUP_SIZE = 25         # completed predictions saved after each group
MEASURE_ENERGY = True
CNN_QUBITS = 5
DNN_QUBITS = 5

DATASETS = [
    "MNIST",
    "FashionMNIST",
    # "KMNIST",
    # "CIFAR100",
    # "TinyImageNet200",
]

MODELS = [
    "CNN",
    "DNN",
    "YOLO26n-CLS",
    "QuantumCNN4",
    "QuantumDNN4",
    "HybridQuantumYOLO",
]

DATA_ROOT = Path("./data")
QUANTUM_CHECKPOINT_DIR = Path("./quantum_models")
HYBRID_QUBITS = 5
HYBRID_LAYERS = 4
# Check these directories if your classical/YOLO checkpoints have moved.
CHECKPOINT_SEARCH_DIRS = [
    Path("."),
    # Path("./classical_checkpoints"),
    # Path("./saved_models"),
    # Path("./saved_model"),
    # Path("./models"),
    QUANTUM_CHECKPOINT_DIR,
]
TINY_IMAGENET_ROOT = Path(os.getenv("TINY_IMAGENET_ROOT", str(DATA_ROOT / "tiny-imagenet-200")))
DEVICE = torch.device("cpu")

# Prefer the same hash as the pre-existing classical script.
if hasattr(classical, "DEVICE_SHORT"):
    SHORT_HASH = str(classical.DEVICE_SHORT)
else:
    hardware_id = f"{socket.gethostname()}-{platform.system()}-{platform.machine()}-{platform.processor()}"
    SHORT_HASH = hashlib.sha256(hardware_id.encode()).hexdigest()[:8]
OUTPUT_ROOT = Path("./test_results") / SHORT_HASH
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

if SAMPLES < 0 or GROUP_SIZE < 1:
    raise ValueError("SAMPLES >= 0 and GROUP_SIZE >= 1 required")

DATASET_CONFIGS = {
    "MNIST": {"key": "mnist", "class": datasets.MNIST, "mean": (0.1307,), "std": (0.3081,), "size": 28, "classes": 10},
    "FashionMNIST": {"key": "fashionmnist", "class": datasets.FashionMNIST, "mean": (0.2860,), "std": (0.3530,), "size": 28, "classes": 10},
    "KMNIST": {"key": "kmnist", "class": datasets.KMNIST, "mean": (0.1918,), "std": (0.3483,), "size": 28, "classes": 10},
    "CIFAR100": {"key": "cifar100", "class": datasets.CIFAR100, "mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761), "size": 32, "classes": 100},
    "TinyImageNet200": {"key": "tinyimagenet200", "class": None, "mean": (0.4802, 0.4481, 0.3975), "std": (0.2302, 0.2265, 0.2262), "size": 64, "classes": 200},
}

# The source quantum script must be the uploaded preprocessing-version-3 code.
if getattr(quantum, "PREPROCESSING_VERSION", None) != 3:
    raise RuntimeError("quantum_dnn_cnn.py must contain the uploaded preprocessing-version-3 implementation")


# ========================== CHECKPOINTS ==========================
def find_one_checkpoint(patterns, roots=CHECKPOINT_SEARCH_DIRS):
    """Find exactly one matching checkpoint; never silently choose an arbitrary file."""
    candidates = {}
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file():
                    candidates[str(path.resolve())] = path
    matches = list(candidates.values())
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for patterns {patterns}; searched {roots}")
    if len(matches) > 1:
        raise RuntimeError("Multiple matching checkpoints; narrow CHECKPOINT_SEARCH_DIRS or filenames:\n" +
                           "\n".join(str(x) for x in matches))
    return matches[0]


def checkpoint_path(model_name, dataset_name, n_qubits=None):
    """Resolve checkpoints using the actual filenames in quantum_models/.

    Names in this folder are fixed; don't silently load a different number
    of qubits or a different model if the requested file is missing.
    """
    key = DATASET_CONFIGS[dataset_name]["key"]
    if model_name in ("QuantumCNN4", "QuantumDNN4"):
        if n_qubits is None or n_qubits < 1:
            raise ValueError("n_qubits must be supplied for quantum CNN/DNN")
        filename = f"{key}_{model_name.lower()}_{n_qubits}qubit_4layer.pth"
    elif model_name == "HybridQuantumYOLO":
        filename = f"{key}_hybrid_yolo26n_{HYBRID_QUBITS}q_{HYBRID_LAYERS}layers.pth"
    else:
        raise ValueError(f"Not a quantum checkpoint model: {model_name}")
    return find_one_checkpoint([filename], roots=[QUANTUM_CHECKPOINT_DIR])


def load_quantum_checkpoint_checked(path, model_name, cfg, n_qubits):
    """Reject incompatible or unversioned saved weights, rather than misreport results."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or "state_dict" not in raw:
        raise ValueError(
            f"{path} does not contain the versioned checkpoint metadata required "
            "by quantum_dnn_cnn.load_checkpoint(). Verify the original training "
            "architecture and preprocessing before using these weights."
        )
    required = ("model_type", "n_qubits", "n_classes", "n_layers", "preprocessing_version")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(
            f"Checkpoint {path} is missing {missing}. Its training preprocessing "
            "cannot be verified against the current inference implementation."
        )
    expected = {
        "model_type": model_name,
        "n_qubits": n_qubits,
        "n_classes": cfg["classes"],
        "n_layers": 4,
        "preprocessing_version": quantum.PREPROCESSING_VERSION,
    }
    for key, val in expected.items():
        if raw[key] != val:
            raise ValueError(f"Incompatible checkpoint {path}: {key}={raw[key]!r}, expected {val!r}")
    return quantum.load_checkpoint(path)



def load_hybrid_checkpoint(path, expected_classes=10,
                           expected_qubits=HYBRID_QUBITS,
                           expected_layers=HYBRID_LAYERS):
    """Reconstruct the exact 5-qubit YOLO head from quantum_yolo(1).py.

    This implementation exposes build_model(n_classes), not build_quantum_yolo()
    or load_checkpoint(). Its QuantumYOLOHead signature is
    (in_features, num_classes, n_qubits=N_QUBITS); the layer count is the
    module-level N_Q_LAYERS, not a constructor keyword.
    """
    if not callable(getattr(quantum_yolo, "build_model", None)):
        raise RuntimeError(
            "Your imported quantum_yolo.py does not define build_model(n_classes). "
            "Use the same quantum_yolo(1).py implementation that trained this checkpoint. "
            f"Imported file: {getattr(quantum_yolo, '__file__', 'unknown')}"
        )
    for key, expected in (("N_QUBITS", expected_qubits),
                          ("N_Q_LAYERS", expected_layers)):
        if not hasattr(quantum_yolo, key) or int(getattr(quantum_yolo, key)) != expected:
            raise RuntimeError(f"quantum_yolo.{key} must be {expected} to match the checkpoint")

    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or not isinstance(raw.get("state_dict"), dict):
        raise ValueError(f"Not a supported hybrid-YOLO state_dict checkpoint: {path}")
    for key, expected in (("n_qubits", expected_qubits),
                          ("n_classes", expected_classes)):
        if key in raw and int(raw[key]) != expected:
            raise ValueError(f"{path}: saved {key}={raw[key]}, expected {expected}")
    for key in ("n_q_layers", "n_layers"):
        if key in raw and int(raw[key]) != expected_layers:
            raise ValueError(f"{path}: saved {key}={raw[key]}, expected {expected_layers}")

    # IMPORTANT: train-time builder constructs the classifier head for n_classes.
    model = quantum_yolo.build_model(expected_classes)
    model.load_state_dict(raw["state_dict"], strict=True)
    return model.to(DEVICE).eval()


def load_saved_model(model_name, cfg, dataset_name):
    key = cfg["key"]
    if model_name in ("CNN", "DNN"):
        stem = f"{key}_{model_name.lower()}"
        path = find_one_checkpoint([f"{stem}.pth", f"**/{stem}.pth"])
        model = classical.SimpleCNN() if model_name == "CNN" else classical.SimpleDNN()
        raw = torch.load(path, map_location="cpu", weights_only=True)
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        model.load_state_dict(state, strict=True)
    elif model_name == "YOLO26n-CLS":
        name = f"yolo26n_{key}_cpu.pt"
        path = find_one_checkpoint([name, f"**/{name}"])

        model, _ = classical_yolo.load_yolo(
            path,
            num_classes=cfg["classes"]
        )
    elif model_name in ("QuantumCNN4", "QuantumDNN4"):
        nq = CNN_QUBITS if model_name == "QuantumCNN4" else DNN_QUBITS
        path = checkpoint_path(model_name, dataset_name, nq)
        model = load_quantum_checkpoint_checked(path, model_name, cfg, nq)
        if (model.model_type != model_name or model.n_qubits != nq or
                model.n_classes != cfg["classes"] or model.n_layers != 4):
            raise ValueError(f"Quantum checkpoint metadata mismatch: {path}")
    elif model_name == "HybridQuantumYOLO":
        path = checkpoint_path(model_name, dataset_name)
        model = load_hybrid_checkpoint(path, expected_classes=cfg["classes"])
        # Check that this model uses the saved 5-qubit, 4-layer architecture.
        if hasattr(model, "n_qubits") and model.n_qubits != HYBRID_QUBITS:
            raise ValueError(f"Hybrid checkpoint qubit mismatch: {path}")
    else:
        raise ValueError(f"Unknown model {model_name}")
    if model_name != "YOLO26n-CLS":
        model = model.to(DEVICE).eval()
    return model, path


def checkpoint_id(path):
    """Hash checkpoint contents to avoid using results from replaced weights."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


# =========================== DATASETS ============================
class TinyImageNetValidation(torch.utils.data.Dataset):
    def __init__(self, root):
        root = Path(root)
        wnids = root / "wnids.txt"
        if not wnids.is_file():
            raise FileNotFoundError(f"Missing {wnids}")
        class_ids = sorted(x.strip() for x in wnids.read_text().splitlines() if x.strip())
        if len(class_ids) != 200:
            raise ValueError("Tiny ImageNet must contain 200 classes")
        class_to_idx = {name: i for i, name in enumerate(class_ids)}
        ann = root / "val" / "val_annotations.txt"
        pairs = []
        if ann.is_file():
            for line in ann.read_text().splitlines():
                fname, class_id, *_ = line.split()
                path = root / "val" / "images" / fname
                if path.is_file() and class_id in class_to_idx:
                    pairs.append((path, class_to_idx[class_id]))
        else:
            for cls, idx in class_to_idx.items():
                for p in sorted((root / "val" / cls).rglob("*")):
                    if p.suffix.lower() in (".png", ".jpeg", ".jpg"):
                        pairs.append((p, idx))
        if not pairs:
            raise FileNotFoundError(f"No labeled Tiny ImageNet validation images in {root}")
        self.samples = sorted(pairs, key=lambda x: str(x[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, y = self.samples[index]
        with Image.open(path) as image:
            return image.convert("RGB"), y


def get_test_dataset(name):
    cfg = DATASET_CONFIGS[name]
    if name == "TinyImageNet200":
        return TinyImageNetValidation(TINY_IMAGENET_ROOT)
    return cfg["class"](root=str(DATA_ROOT), train=False, download=True, transform=None)


# ======================== MODEL INFERENCE ========================
def prepare_input(model_name, image, cfg):
    if model_name == "YOLO26n-CLS":
        return image  # original YOLO wrapper controls its preprocessing
    if model_name == "HybridQuantumYOLO":
        # Match quantum_yolo(1).py get_dataset preprocessing (resize, RGB, tensor).
        size = int(quantum_yolo.IMG_SIZE)
        image = transforms.Resize((size, size))(image).convert("RGB")
        return transforms.ToTensor()(image).unsqueeze(0).to(DEVICE)
    # Matches tensor shape passed by the uploaded quantum training code.
    return transforms.ToTensor()(transforms.Resize((cfg["size"], cfg["size"]))(image))


def infer(model_name, model, prepared, cfg):
    with torch.no_grad():
        if model_name in ("CNN", "DNN"):
            pred, logits = classical._run_torch_model(model, prepared, mean=cfg["mean"], std=cfg["std"])
        elif model_name == "YOLO26n-CLS":
            pred, logits = classical_yolo.run_yolo_inference(model, prepared)
        elif model_name in ("QuantumCNN4", "QuantumDNN4"):
            # Uses the exact uploaded version-3 quantum preprocessing.
            pred, logits = quantum.run_quantum_inference(model, prepared, mean=cfg["mean"], std=cfg["std"])
        elif model_name == "HybridQuantumYOLO":
            logits = quantum_yolo.get_logits(model(prepared), cfg["classes"])
            pred = int(logits.argmax(dim=1).item())
        else:
            raise ValueError(model_name)
    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)
    logits = logits.detach().float().reshape(-1)
    if logits.numel() != cfg["classes"]:
        raise ValueError(f"{model_name}: expected {cfg['classes']} logits, got {logits.numel()}")
    return int(pred), logits


def quality(logits):
    probs = torch.softmax(logits, dim=0)
    top2 = torch.topk(logits, k=2).values
    return {
        "confidence_score": float(probs.max().item()),
        "logit_margin": float((top2[0] - top2[1]).item()),
        "entropy": float(-(probs * torch.log(probs + 1e-12)).sum().item()),
    }


# ====================== ENERGY / CSV HELPERS =====================
def measure_group(model_name, model, prepared_entries, cfg):
    """Energy is measured over whole group then amortized per forward pass."""
    tracker = None
    if MEASURE_ENERGY and EmissionsTracker is not None:
        tracker = EmissionsTracker(project_name="unified_inference_telemetry",
                                   save_to_file=False, log_level="error", measure_power_secs=1)
        tracker.start()
    results = []
    try:
        for index, label, prepared in prepared_entries:
            before_mb = psutil.Process().memory_info().rss / (1024 ** 2)
            start = time.perf_counter()
            pred, logits = infer(model_name, model, prepared, cfg)
            elapsed = time.perf_counter() - start
            results.append({"sample_index": index,
                            "true_label": int(label), "prediction": pred,
                            "correct": int(pred == label),
                            "execution_time_sec": elapsed,
                            "memory_before_mb": before_mb,
                            "memory_footprint_mb": psutil.Process().memory_info().rss / (1024 ** 2),
                            **quality(logits)})
    finally:
        emissions = tracker.stop() if tracker is not None else None
    data = getattr(tracker, "final_emissions_data", None) if tracker is not None else None
    n = len(results)
    energy = {}
    for col, attr in [("cpu_energy_kwh", "cpu_energy"),
                      ("gpu_energy_kwh", "gpu_energy"),
                      ("ram_energy_kwh", "ram_energy"),
                      ("total_energy_kwh", "energy_consumed")]:
        value = getattr(data, attr, None) if data is not None else None
        energy[col] = float(value) / n if value is not None and n else None
    energy["total_emissions_kg"] = float(emissions) / n if emissions is not None and n else None
    for row in results:
        row.update(energy)
        row.update({"timestamp": datetime.now(timezone.utc).isoformat(),
                    "cpu_usage_pct": psutil.cpu_percent(interval=None),
                    "cpu_clock_mhz": (freq.current if (freq := psutil.cpu_freq()) else None),
                    "ram_usage_pct": psutil.virtual_memory().percent})
    return results


def load_existing(path, model_name, expected_digest, count):
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"sample_index", "model_type", "checkpoint_sha256_12",
                "true_label", "prediction"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Existing file {path} misses {sorted(missing)}; archive it before rerunning")
    if df[list(required)].isna().any().any():
        raise ValueError(f"Incomplete/malformed CSV: {path}")
    if set(df["model_type"].astype(str)) != {model_name}:
        raise ValueError(f"Model identity mismatch in {path}")
    if set(df["checkpoint_sha256_12"].astype(str)) != {expected_digest}:
        raise ValueError(f"Checkpoint has changed since {path} was created; archive old CSV")
    if "repeat_index" in df.columns:
        raise ValueError(f"Existing CSV {path} uses the older repeated-inference format; archive it before running once per sample")
    if df["sample_index"].duplicated().any():
        raise ValueError(f"Duplicate sample indices: {path}")
    if ((df["sample_index"] < 0) | (df["sample_index"] >= count)).any():
        raise ValueError(f"Sample indices disagree with current SAMPLES: {path}")
    return df


def atomic_save(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def metrics_from_df(df):
    # Exactly one inference and prediction per image.
    yt = df["true_label"].astype(int)
    yp = df["prediction"].astype(int)
    return {
        "model_accuracy": accuracy_score(yt, yp),
        "model_precision_weighted": precision_score(yt, yp, average="weighted", zero_division=0),
        "model_recall_weighted": recall_score(yt, yp, average="weighted", zero_division=0),
        "model_f1_weighted": f1_score(yt, yp, average="weighted", zero_division=0),
    }


def refresh_dataset_summary(dataset_name):
    """Rebuild combined CSVs from ALL existing model CSVs, including skipped runs."""
    folder = OUTPUT_ROOT / dataset_name
    # Only use per-model CSVs created by this script, not unrelated older CSVs.
    model_files = sorted(folder.glob("*.csv"))
    if not model_files:
        return
    frames, summary_rows = [], []
    for path in model_files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        required = {"model_type", "execution_class", "sample_index", "true_label",
                    "prediction", "checkpoint_sha256_12"}
        if not required.issubset(df.columns) or "repeat_index" in df.columns:
            print(f"[SKIP unrelated/legacy CSV in summary] {path}")
            continue
        if df["sample_index"].duplicated().any():
            raise ValueError(f"Duplicate sample indices in summary source: {path}")
        frames.append(df)
        model_name = str(df["model_type"].iloc[0])
        m = metrics_from_df(df)
        summary_rows.append({"dataset": dataset_name, "model_type": model_name,
                             "execution_class": df["execution_class"].iloc[0],
                             "n_samples": int(df["sample_index"].nunique()),
                             "n_inferences": len(df),
                             "checkpoint_sha256_12": df["checkpoint_sha256_12"].iloc[0], **m})
    if frames:
        atomic_save(pd.concat(frames, ignore_index=True), OUTPUT_ROOT / f"{dataset_name.lower()}_energy_telemetry.csv")
        atomic_save(pd.DataFrame(summary_rows), OUTPUT_ROOT / f"{dataset_name.lower()}_energy_summary.csv")


# ============================= MAIN ==============================
def run():
    print(f"Using quantum_yolo module: {getattr(quantum_yolo, '__file__', 'unknown')}")
    print(f"Results: {OUTPUT_ROOT.resolve()}")
    print(f"Datasets: {DATASETS}; samples={SAMPLES or 'ALL'}; one inference per image")
    if MEASURE_ENERGY and EmissionsTracker is None:
        print("CodeCarbon not installed: energy columns will be missing (not zero).")
    for dataset_name in DATASETS:
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        cfg = DATASET_CONFIGS[dataset_name]
        ds = get_test_dataset(dataset_name)
        count = len(ds) if SAMPLES == 0 else min(SAMPLES, len(ds))
        output_dir = OUTPUT_ROOT / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for model_name in MODELS:
            execution_type = ("quantum" if model_name.startswith("Quantum") or
                              model_name == "HybridQuantumYOLO" else "classical")
            print(f"\n[{dataset_name}] {model_name}")
            try:
                model, checkpoint_path = load_saved_model(model_name, cfg, dataset_name)
            except FileNotFoundError as exc:
                print(f"[SKIP missing checkpoint] {exc}")
                continue
            digest = checkpoint_id(checkpoint_path)
            path = output_dir / f"{model_name}_{execution_type}_{count}.csv"
            df = load_existing(path, model_name, digest, count)
            done = set(df["sample_index"].astype(int)) if not df.empty else set()
            pending = [i for i in range(count) if i not in done]
            print(f"Done {len(done)}/{count}; remaining {len(pending)}")
            if pending:
                if model_name == "YOLO26n-CLS":
                    underlying = model.model
                else:
                    underlying = model
                parameters = sum(p.numel() for p in underlying.parameters())
                # Warmup is not included in timing or energy.
                image, _ = ds[0]
                infer(model_name, model, prepare_input(model_name, image, cfg), cfg)
                for offset in tqdm(range(0, len(pending), GROUP_SIZE), desc=model_name):
                    batch = pending[offset:offset + GROUP_SIZE]
                    prepared_entries = []
                    for i in batch:
                        img, label = ds[i]
                        prepared = prepare_input(model_name, img, cfg)
                        prepared_entries.append((i, label, prepared))
                    new_rows = measure_group(model_name, model, prepared_entries, cfg)
                    for row in new_rows:
                        row.update({"dataset": dataset_name, "model_type": model_name,
                                    "execution_class": execution_type,
                                    "quantum_computing": execution_type == "quantum",
                                    "quantum_execution_mode": ("classical_simulation" if execution_type == "quantum" else "not_applicable"),
                                    "n_qubits": (int(model.n_qubits) if model_name.startswith("Quantum") else None),
                                    "checkpoint_path": str(checkpoint_path.resolve()),
                                    "checkpoint_sha256_12": digest,
                                    "parameters": parameters,
                                    "device_type": "cpu", "cpu_model": platform.processor(),
                                    "pc_name": socket.gethostname(), "os_name": platform.system()})
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                    atomic_save(df, path)
            # Backfill metrics for newly completed AND previously skipped files.
            if not df.empty:
                metrics = metrics_from_df(df)
                for key, val in metrics.items():
                    df[key] = val
                atomic_save(df, path)
                print(f"Saved {path}; accuracy={metrics['model_accuracy']:.4f}")
            del model
        refresh_dataset_summary(dataset_name)
    print("\nTelemetry collection completed.")


if __name__ == "__main__":
    run()
