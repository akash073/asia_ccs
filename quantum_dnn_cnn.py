# ============================================================
# 4-LAYER QUANTUM DNN + 4-LAYER QUANTUM CNN
# SAME TELEMETRY SCHEMA AS THE UPLOADED CLASSICAL CNN/DNN CODE
#
# Key difference:
#     quantum_computing = True
#
# Quantum-only fields are populated:
#     source_qubits
#     n_qubits
#     fidelity
#     pennylane_device
#     pennylane_version
#
# Behavior:
#   - 4 qubits by default (configurable)
#   - exactly 4 quantum layers
#   - train if checkpoint does not exist
#   - load saved checkpoint if available
#   - train for 10 epochs by default
#   - collect the same runtime/system telemetry columns
#   - save one CSV per model/dataset
#   - backfill Accuracy / Precision / Recall / F1
# ============================================================

from __future__ import annotations

import hashlib
import os
import platform
import socket
import sys
import time
from pathlib import Path
from PIL import Image
from typing import Iterable

import numpy as np
import pandas as pd
import pennylane as qml
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = int(os.getenv("RANDOM_SEED", "42"))
np.random.seed(SEED)
torch.manual_seed(SEED)

NUM_TEST_SAMPLES = int(
    os.getenv("NUM_TEST_SAMPLES", "10")
)

TRAIN_SAMPLES = int(
    os.getenv("TRAIN_SAMPLES", "60000")
)

TRAIN_EPOCHS = int(
    os.getenv("TRAIN_EPOCHS", "10")
)

TRAIN_BATCH_SIZE = int(
    os.getenv("TRAIN_BATCH_SIZE", "16")
)

LEARNING_RATE = float(
    os.getenv("LEARNING_RATE", "0.001")
)

FLUSH_EVERY = int(
    os.getenv("FLUSH_EVERY", "25")
)

DNN_QUBITS = int(
    os.getenv("DNN_QUBITS", "4")
)

CNN_QUBITS = int(
    os.getenv("CNN_QUBITS", "4")
)

N_LAYERS = 4
N_CLASSES = 10

PENNYLANE_DEVICE = os.getenv(
    "PENNYLANE_DEVICE",
    "default.qubit",
)

FORCE_RETRAIN = (
    os.getenv("FORCE_RETRAIN", "0").strip() == "1"
)

DEVICE_MODE = os.getenv(
    "DEVICE_MODE",
    "cpu",
).lower()


def resolve_device():
    if DEVICE_MODE == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")

        print(
            "CUDA requested but unavailable. "
            "Falling back to CPU."
        )

        return torch.device("cpu")

    if DEVICE_MODE == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device("cpu")


# PennyLane default.qubit is CPU-based. Keep PyTorch on CPU by default.
DEVICE = resolve_device()

DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data"))
OUTPUT_ROOT = Path.cwd() / "test_results"
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", str(Path.cwd() / "quantum_checkpoints")))

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Class mappings match torchvision ImageFolder's sorted class IDs.
DATASET_CONFIGS = {
    "MNIST": {"dataset_class": datasets.MNIST,
              "mean": (0.1307,), "std": (0.3081,),
              "checkpoint_prefix": "mnist", "num_classes": 10,
              "image_size": 28, "channels": 1},
    "FashionMNIST": {"dataset_class": datasets.FashionMNIST,
              "mean": (0.2860,), "std": (0.3530,),
              "checkpoint_prefix": "fashionmnist", "num_classes": 10,
              "image_size": 28, "channels": 1},
    "KMNIST": {"dataset_class": datasets.KMNIST,
              "mean": (0.1918,), "std": (0.3483,),
              "checkpoint_prefix": "kmnist", "num_classes": 10,
              "image_size": 28, "channels": 1},
    "CIFAR100": {"dataset_class": datasets.CIFAR100,
              "mean": (0.5071, 0.4867, 0.4408),
              "std": (0.2675, 0.2565, 0.2761),
              "checkpoint_prefix": "cifar100", "num_classes": 100,
              "image_size": 32, "channels": 3},
    "TinyImageNet200": {"dataset_class": None,
              "mean": (0.4802, 0.4481, 0.3975),
              "std": (0.2302, 0.2265, 0.2262),
              "checkpoint_prefix": "tinyimagenet200", "num_classes": 200,
              "image_size": 64, "channels": 3},
}

# A new checkpoint filename prevents loading weights learned with the original
# degenerate 1-qubit min/max preprocessing.
PREPROCESSING_VERSION = 2

class TinyImageNetValidation(torch.utils.data.Dataset):
    """Original Tiny ImageNet 200 validation layout (with annotations)."""
    def __init__(self, root, transform=None, class_to_idx=None):
        root = Path(root)
        self.transform = transform
        wnids_file = root / "wnids.txt"
        if not wnids_file.is_file():
            raise FileNotFoundError(f"Missing Tiny ImageNet wnids.txt: {wnids_file}")
        ids = sorted(x.strip() for x in wnids_file.read_text().splitlines() if x.strip())
        if len(ids) != 200:
            raise ValueError(f"Expected 200 Tiny ImageNet classes, found {len(ids)}")
        self.class_to_idx = class_to_idx or {x: i for i, x in enumerate(ids)}
        ann = root / "val" / "val_annotations.txt"
        self.samples = []
        if ann.is_file():
            for line in ann.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    file_name, label = parts[:2]
                    image_path = root / "val" / "images" / file_name
                    if label in self.class_to_idx and image_path.is_file():
                        self.samples.append((image_path, self.class_to_idx[label]))
        else:
            for label, idx in self.class_to_idx.items():
                for path in sorted((root / "val" / label).rglob("*")):
                    if path.suffix.lower() in (".jpeg", ".jpg", ".png"):
                        self.samples.append((path, idx))
        if not self.samples:
            raise FileNotFoundError(f"No labeled validation images found in {root / 'val'}")
        self.samples.sort(key=lambda v: str(v[0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as im:
            image = im.convert("RGB")
            return (self.transform(image) if self.transform else image), label


def load_dataset_pair(dataset_name, config):
    """Train split for fitting; unseen official test/val split for telemetry."""
    shape_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
    ])
    if dataset_name != "TinyImageNet200":
        ds = config["dataset_class"]
        train = ds(root=str(DATA_ROOT), train=True, download=True,
                   transform=shape_transform)
        test = ds(root=str(DATA_ROOT), train=False, download=True,
                  transform=shape_transform)
        return train, test

    root = Path(os.getenv("TINY_IMAGENET_ROOT", str(DATA_ROOT / "tiny-imagenet-200")))
    train_root = root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(
            f"Missing Tiny ImageNet training directory: {train_root}. "
            "Download and extract tiny-imagenet-200 manually."
        )
    train = datasets.ImageFolder(str(train_root), transform=shape_transform)
    if len(train.classes) != 200:
        raise ValueError(f"Tiny ImageNet train contains {len(train.classes)} classes, expected 200")
    test = TinyImageNetValidation(root, transform=shape_transform,
                                  class_to_idx=train.class_to_idx)
    return train, test


QUANTUM_MODELS = (
    "QuantumCNN4",
    "QuantumDNN4",
)


# ============================================================
# 2. OPTIONAL PACKAGES
# ============================================================

try:
    from codecarbon import EmissionsTracker
    import codecarbon

    CODECARBON_AVAILABLE = True
    CODECARBON_VERSION = str(
        codecarbon.__version__
    )

except Exception:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False
    CODECARBON_VERSION = "unavailable"

    print(
        "CodeCarbon not available. "
        "Energy values will be set to 0."
    )


try:
    import pynvml

    pynvml.nvmlInit()

    NVML_AVAILABLE = True

    NVML_HANDLE = (
        pynvml.nvmlDeviceGetHandleByIndex(0)
        if torch.cuda.is_available()
        else None
    )

except Exception:
    NVML_AVAILABLE = False
    NVML_HANDLE = None

    print(
        "pynvml not available."
    )


try:
    import cpuinfo

    _CPU_INFO = (
        cpuinfo.get_cpu_info()
    )

    CPU_MODEL_NAME = (
        _CPU_INFO.get(
            "brand_raw",
            "Unknown",
        )
    )

    CPU_ARCH = (
        _CPU_INFO.get(
            "arch",
            platform.machine(),
        )
    )

except Exception:
    CPU_MODEL_NAME = (
        platform.processor()
        or "Unknown"
    )

    CPU_ARCH = (
        platform.machine()
    )


try:
    from fvcore.nn import FlopCountAnalysis

    FVCORE_AVAILABLE = True

except Exception:
    FlopCountAnalysis = None
    FVCORE_AVAILABLE = False

    print(
        "fvcore not available."
    )


CPU_TDP_W = None


# ============================================================
# 3. SYSTEM INFORMATION
# ============================================================

TORCH_VERSION = str(
    torch.__version__
)

PYTHON_VERSION = (
    sys.version.split()[0]
)

PENNYLANE_VERSION = str(
    qml.__version__
)

OS_NAME = platform.system()
OS_VERSION = platform.version()
OS_ARCHITECTURE = platform.machine()

SYSTEM_RAM_TOTAL_GB = round(
    psutil.virtual_memory().total
    / (1024 ** 3),
    2,
)

CPU_CORE_COUNT = psutil.cpu_count(
    logical=False
)

CPU_THREAD_COUNT = psutil.cpu_count(
    logical=True
)


def get_os_full_name():

    system = platform.system()
    architecture = platform.machine()

    if system == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                (
                    r"SOFTWARE\Microsoft"
                    r"\Windows NT"
                    r"\CurrentVersion"
                ),
            )

            product_name = (
                winreg.QueryValueEx(
                    key,
                    "ProductName",
                )[0]
            )

            display_version = (
                winreg.QueryValueEx(
                    key,
                    "DisplayVersion",
                )[0]
            )

            current_build = (
                winreg.QueryValueEx(
                    key,
                    "CurrentBuild",
                )[0]
            )

            return (
                f"{product_name} "
                f"{display_version} "
                f"Build {current_build} "
                f"{architecture}"
            )

        except Exception:
            return (
                f"Windows "
                f"{platform.release()} "
                f"{architecture}"
            )

    if system == "Linux":
        info = {}

        try:
            with open(
                "/etc/os-release",
                "r",
                encoding="utf-8",
            ) as f:
                for line in f:
                    if "=" in line:
                        k, v = (
                            line.strip()
                            .split("=", 1)
                        )
                        info[k] = (
                            v.strip('"')
                        )

        except Exception:
            pass

        pretty_name = (
            info.get(
                "PRETTY_NAME"
            )
        )

        if pretty_name:
            return (
                f"{pretty_name} "
                f"{architecture}"
            )

        return (
            f"Linux "
            f"{platform.release()} "
            f"{architecture}"
        )

    if system == "Darwin":
        return (
            f"macOS "
            f"{platform.mac_ver()[0]} "
            f"{architecture}"
        )

    return (
        f"{system} "
        f"{platform.release()} "
        f"{architecture}"
    )


OS_FULL_NAME = (
    get_os_full_name()
)


# ============================================================
# 4. DEVICE ID
# ============================================================

def make_stable_device_id():

    raw = (
        f"{socket.gethostname()}-"
        f"{platform.system()}-"
        f"{platform.machine()}-"
        f"{CPU_MODEL_NAME}"
    )

    return hashlib.sha256(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()


DEVICE_UUID = (
    make_stable_device_id()
)

DEVICE_SHORT = (
    DEVICE_UUID[:8]
)

DEVICE_LOG_DIR = (
    OUTPUT_ROOT
    / DEVICE_SHORT
)

DEVICE_LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def get_hostname():
    return socket.gethostname()


# ============================================================
# 5. GPU STATIC INFORMATION
# ============================================================

def get_cuda_driver_version():

    if not NVML_AVAILABLE:
        return None

    try:
        value = (
            pynvml
            .nvmlSystemGetDriverVersion()
        )

        if isinstance(
            value,
            bytes,
        ):
            return value.decode(
                "utf-8"
            )

        return value

    except Exception:
        return None


CUDA_DRIVER_VERSION = (
    get_cuda_driver_version()
)


def get_gpu_static():

    result = {
        "gpu_driver_version":
            CUDA_DRIVER_VERSION,

        "gpu_compute_capability":
            None,

        "gpu_power_limit_w":
            None,

        "gpu_memory_total_mb":
            None,
    }

    if torch.cuda.is_available():
        try:
            props = (
                torch.cuda
                .get_device_properties(0)
            )

            result[
                "gpu_compute_capability"
            ] = (
                f"{props.major}."
                f"{props.minor}"
            )

        except Exception:
            pass

    if (
        not NVML_AVAILABLE
        or NVML_HANDLE is None
    ):
        return result

    try:
        power_limit = (
            pynvml
            .nvmlDeviceGetPowerManagementLimit(
                NVML_HANDLE
            )
        )

        memory = (
            pynvml
            .nvmlDeviceGetMemoryInfo(
                NVML_HANDLE
            )
        )

        result[
            "gpu_power_limit_w"
        ] = round(
            power_limit / 1000.0,
            2,
        )

        result[
            "gpu_memory_total_mb"
        ] = round(
            memory.total
            / (1024 ** 2),
            2,
        )

    except Exception:
        pass

    return result


GPU_STATIC = (
    get_gpu_static()
)


def get_gpu_core_thread():

    if not torch.cuda.is_available():
        return (
            None,
            None,
        )

    try:
        props = (
            torch.cuda
            .get_device_properties(0)
        )

        sm_count = (
            props.multi_processor_count
        )

        cores_per_sm = {
            2: 32,
            3: 192,
            5: 128,
            6: 64,
            7: 64,
            8: 128,
            9: 128,
        }.get(
            props.major,
            64,
        )

        gpu_core_count = (
            sm_count
            * cores_per_sm
        )

        gpu_thread_count = (
            sm_count
            * props.max_threads_per_multi_processor
        )

        return (
            gpu_core_count,
            gpu_thread_count,
        )

    except Exception:
        return (
            None,
            None,
        )


(
    GPU_CORE_COUNT,
    GPU_THREAD_COUNT,
) = get_gpu_core_thread()


# ============================================================
# 6. RUNTIME HARDWARE HELPERS
# ============================================================

def get_gpu_name():

    try:
        if torch.cuda.is_available():
            return (
                torch.cuda
                .get_device_name(0)
            )

        return "No GPU"

    except Exception:
        return "Unknown"


def get_cpu_usage():

    try:
        return (
            psutil.cpu_percent(
                interval=None
            )
        )

    except Exception:
        return None


def get_ram_usage():

    try:
        return (
            psutil
            .virtual_memory()
            .percent
        )

    except Exception:
        return None


def get_cpu_freq():

    try:
        freq = (
            psutil.cpu_freq()
        )

        return (
            round(
                freq.current,
                2,
            )
            if freq
            else None
        )

    except Exception:
        return None


def get_memory_footprint_mb():

    try:
        return round(
            psutil
            .Process(
                os.getpid()
            )
            .memory_info()
            .rss
            / (1024 ** 2),
            4,
        )

    except Exception:
        return None


def get_cpu_temp():

    try:
        temps = (
            psutil
            .sensors_temperatures()
        )

        if not temps:
            return None

        for key in (
            "coretemp",
            "k10temp",
            "cpu_thermal",
            "acpitz",
        ):
            if key in temps:
                values = [
                    item.current
                    for item
                    in temps[key]
                    if (
                        item.current
                        is not None
                        and item.current > 0
                    )
                ]

                if values:
                    return round(
                        sum(values)
                        / len(values),
                        1,
                    )

    except Exception:
        pass

    return None


def get_cpu_power_draw_w():
    return None


def get_cpu_cores_used():

    try:
        return sum(
            1
            for value
            in psutil.cpu_percent(
                percpu=True
            )
            if value > 1.0
        )

    except Exception:
        return None


def get_gpu_metrics():

    result = {
        "gpu_power_draw_w":
            None,

        "gpu_utilization_pct":
            None,

        "gpu_temp_c":
            None,

        "gpu_memory_used_mb":
            None,

        "gpu_sm_clock_mhz":
            None,

        "gpu_memory_clock_mhz":
            None,
    }

    if (
        not NVML_AVAILABLE
        or NVML_HANDLE is None
    ):
        return result

    try:
        power = (
            pynvml
            .nvmlDeviceGetPowerUsage(
                NVML_HANDLE
            )
        )

        utilization = (
            pynvml
            .nvmlDeviceGetUtilizationRates(
                NVML_HANDLE
            )
        )

        temperature = (
            pynvml
            .nvmlDeviceGetTemperature(
                NVML_HANDLE,
                pynvml.NVML_TEMPERATURE_GPU,
            )
        )

        memory = (
            pynvml
            .nvmlDeviceGetMemoryInfo(
                NVML_HANDLE
            )
        )

        sm_clock = (
            pynvml
            .nvmlDeviceGetClockInfo(
                NVML_HANDLE,
                pynvml.NVML_CLOCK_SM,
            )
        )

        memory_clock = (
            pynvml
            .nvmlDeviceGetClockInfo(
                NVML_HANDLE,
                pynvml.NVML_CLOCK_MEM,
            )
        )

        return {
            "gpu_power_draw_w":
                round(
                    power / 1000.0,
                    2,
                ),

            "gpu_utilization_pct":
                utilization.gpu,

            "gpu_temp_c":
                temperature,

            "gpu_memory_used_mb":
                round(
                    memory.used
                    / (1024 ** 2),
                    2,
                ),

            "gpu_sm_clock_mhz":
                sm_clock,

            "gpu_memory_clock_mhz":
                memory_clock,
        }

    except Exception:
        return result


# ============================================================
# 7. SAME REFERENCE COLUMN ORDER AS UPLOADED CODE
# ============================================================

REFERENCE_COLUMNS = [
    "timestamp",
    "unique_device_id",
    "device_short_id",
    "pc_name",
    "collection_mode",

    "sample_index",
    "sample_id",
    "true_label",
    "prediction",
    "correct",

    "dataset",
    "model_type",
    "parameters",
    "model_flops",
    "checkpoint_path",

    "confidence_score",
    "logit_margin",
    "entropy",

    "execution_time_sec",

    "cpu_energy_kwh",
    "gpu_energy_kwh",
    "ram_energy_kwh",
    "total_energy_kwh",
    "total_emissions_kg",
    "carbon_intensity_kgco2_kwh",
    "codecarbon_version",

    "input_tokens",
    "output_tokens",
    "total_tokens",
    "tokens_per_second",
    "joules_per_token",
    "energy_per_token_kwh",
    "watts_estimated",
    "gpu_energy_pct_of_total",
    "cpu_energy_pct_of_total",

    "cpu_model",
    "cpu_architecture",
    "cpu_core_count",
    "cpu_thread_count",
    "cpu_core",
    "cpu_thread",
    "cpu_tdp_w",
    "cpu_usage_pct",
    "cpu_clock_mhz",
    "cpu_temp_c",
    "cpu_power_draw_w",
    "cpu_cores_used",

    "gpu_model",
    "gpu_core",
    "gpu_thread",
    "gpu_driver_version",
    "gpu_compute_capability",
    "gpu_power_limit_w",
    "gpu_memory_total_mb",
    "gpu_power_draw_w",
    "gpu_utilization_pct",
    "gpu_temp_c",
    "gpu_memory_used_mb",
    "gpu_sm_clock_mhz",
    "gpu_memory_clock_mhz",
    "cuda_driver_version",
    "cuda_available",
    "device_type",

    "ram_usage_pct",
    "memory_footprint_mb",
    "system_ram_total_gb",

    "os_name",
    "os_version",
    "os_architecture",
    "os_full_name",
    "python_version",
    "torch_version",

    "model_accuracy",
    "model_precision_weighted",
    "model_recall_weighted",
    "model_f1_weighted",

    "quantum_computing",
    "model_under_attack",

    "source_qubits",
    "n_qubits",
    "fidelity",
    "pennylane_device",
    "pennylane_version",
]


# ============================================================
# 8. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_tensor, n_qubits, mean, std):
    """Compress normalized image to bounded rotation angles.

    tanh preserves magnitude for n_qubits=1; unlike per-image min/max,
    a single-qubit input is not identically zero. Applies equally to RGB.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    image = transforms.Normalize(mean, std)(image_tensor.float())
    flat = image.reshape(-1)
    if n_qubits > flat.numel():
        raise ValueError(f"n_qubits={n_qubits} exceeds pixels={flat.numel()}")
    chunks = torch.tensor_split(flat, n_qubits)
    compressed = torch.stack([chunk.mean() for chunk in chunks])
    return torch.pi * torch.tanh(compressed)


# ============================================================
# 9. FOUR-LAYER QUANTUM DNN
# ============================================================

def build_dnn_4layer_qnode(
    n_qubits,
    device_name,
):

    dev = qml.device(
        device_name,
        wires=n_qubits,
    )

    @qml.qnode(
        dev,
        interface="torch",
        diff_method="best",
    )
    def circuit(
        x,
        weights,
    ):

        # Input encoding
        for q in range(
            n_qubits
        ):
            qml.RY(
                x[q],
                wires=q,
            )

            qml.RZ(
                x[q],
                wires=q,
            )

        # ------------------------
        # Layer 1
        # ------------------------
        for q in range(
            n_qubits
        ):
            qml.Rot(
                weights[
                    0,
                    q,
                    0,
                ],
                weights[
                    0,
                    q,
                    1,
                ],
                weights[
                    0,
                    q,
                    2,
                ],
                wires=q,
            )

        if n_qubits > 1:
            for q in range(
                n_qubits - 1
            ):
                qml.CNOT(
                    wires=[
                        q,
                        q + 1,
                    ]
                )

        # ------------------------
        # Layer 2
        # ------------------------
        for q in range(
            n_qubits
        ):
            qml.Rot(
                weights[
                    1,
                    q,
                    0,
                ],
                weights[
                    1,
                    q,
                    1,
                ],
                weights[
                    1,
                    q,
                    2,
                ],
                wires=q,
            )

        if n_qubits > 1:
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])

        # ------------------------
        # Layer 3
        # ------------------------
        for q in range(
            n_qubits
        ):
            qml.Rot(
                weights[
                    2,
                    q,
                    0,
                ],
                weights[
                    2,
                    q,
                    1,
                ],
                weights[
                    2,
                    q,
                    2,
                ],
                wires=q,
            )

        if n_qubits > 1:
            for q in range(
                n_qubits - 1
            ):
                qml.CZ(
                    wires=[
                        q,
                        q + 1,
                    ]
                )

        # ------------------------
        # Layer 4
        # ------------------------
        for q in range(
            n_qubits
        ):
            qml.Rot(
                weights[
                    3,
                    q,
                    0,
                ],
                weights[
                    3,
                    q,
                    1,
                ],
                weights[
                    3,
                    q,
                    2,
                ],
                wires=q,
            )

        return [
            qml.expval(
                qml.PauliZ(q)
            )
            for q
            in range(
                n_qubits
            )
        ]

    return circuit


# ============================================================
# 10. FOUR-LAYER QUANTUM CNN
# ============================================================

def build_cnn_4layer_qnode(
    n_qubits,
    device_name,
):

    dev = qml.device(
        device_name,
        wires=n_qubits,
    )

    @qml.qnode(
        dev,
        interface="torch",
        diff_method="best",
    )
    def circuit(
        x,
        weights,
    ):

        # Input encoding
        for q in range(
            n_qubits
        ):
            qml.Hadamard(
                wires=q
            )

            qml.RY(
                x[q],
                wires=q,
            )

        # ------------------------
        # Layer 1
        # ------------------------
        for q in range(
            n_qubits
        ):
            qml.Rot(
                weights[
                    0,
                    q,
                    0,
                ],
                weights[
                    0,
                    q,
                    1,
                ],
                weights[
                    0,
                    q,
                    2,
                ],
                wires=q,
            )

        if n_qubits > 1:
            for q in range(
                0,
                n_qubits - 1,
                2,
            ):
                qml.CNOT(
                    wires=[
                        q,
                        q + 1,
                    ]
                )

        # ------------------------
        # Layer 2
        # ------------------------
        for q in range(
            n_qubits
        ):
            qml.Rot(
                weights[
                    1,
                    q,
                    0,
                ],
                weights[
                    1,
                    q,
                    1,
                ],
                weights[
                    1,
                    q,
                    2,
                ],
                wires=q,
            )

        if n_qubits > 2:
            for q in range(
                1,
                n_qubits - 1,
                2,
            ):
                qml.CZ(
                    wires=[
                        q,
                        q + 1,
                    ]
                )

        # ------------------------
        # Layer 3
        # ------------------------
        for q in range(
            n_qubits
        ):
            qml.Rot(
                weights[
                    2,
                    q,
                    0,
                ],
                weights[
                    2,
                    q,
                    1,
                ],
                weights[
                    2,
                    q,
                    2,
                ],
                wires=q,
            )

        if n_qubits > 1:
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])

        # ------------------------
        # Layer 4
        # ------------------------
        for q in range(
            n_qubits
        ):
            qml.Rot(
                weights[
                    3,
                    q,
                    0,
                ],
                weights[
                    3,
                    q,
                    1,
                ],
                weights[
                    3,
                    q,
                    2,
                ],
                wires=q,
            )

        return [
            qml.expval(
                qml.PauliZ(q)
            )
            for q
            in range(
                n_qubits
            )
        ]

    return circuit


# ============================================================
# 11. TRAINABLE QUANTUM CLASSIFIER
# ============================================================

class Quantum4LayerClassifier(
    nn.Module
):

    def __init__(
        self,
        model_type,
        n_qubits,
        n_classes=N_CLASSES,
        device_name=PENNYLANE_DEVICE,
    ):
        super().__init__()

        self.model_type = (
            model_type
        )

        self.n_qubits = (
            n_qubits
        )

        self.n_layers = 4

        self.n_classes = (
            n_classes
        )

        self.device_name = (
            device_name
        )

        if (
            model_type
            == "QuantumDNN4"
        ):
            self.qnode = (
                build_dnn_4layer_qnode(
                    n_qubits,
                    device_name,
                )
            )

        elif (
            model_type
            == "QuantumCNN4"
        ):
            self.qnode = (
                build_cnn_4layer_qnode(
                    n_qubits,
                    device_name,
                )
            )

        else:
            raise ValueError(
                f"Unknown model type: "
                f"{model_type}"
            )

        self.q_weights = nn.Parameter(
            0.01
            * torch.randn(
                4,
                n_qubits,
                3,
            )
        )

        self.readout = nn.Linear(
            n_qubits,
            n_classes,
        )

    def _single(
        self,
        x,
    ):

        output = self.qnode(
            x,
            self.q_weights,
        )

        if isinstance(
            output,
            torch.Tensor,
        ):
            q_features = (
                output.float()
            )

        else:
            q_features = (
                torch.stack([
                    value
                    if isinstance(
                        value,
                        torch.Tensor,
                    )
                    else torch.tensor(
                        value,
                        dtype=torch.float32,
                    )
                    for value
                    in output
                ])
                .float()
            )

        return self.readout(
            q_features
        )

    def forward(
        self,
        x,
    ):

        if x.ndim == 1:
            return (
                self._single(
                    x
                )
                .unsqueeze(0)
            )

        return torch.stack([
            self._single(
                sample
            )
            for sample
            in x
        ])

    @property
    def parameter_count(
        self,
    ):

        return int(
            sum(
                p.numel()
                for p
                in self.parameters()
            )
        )


# ============================================================
# 12. CHECKPOINT SAVE / LOAD
# ============================================================

def get_checkpoint_path(
    model_name,
    dataset_name,
    n_qubits,
):

    return (
        CHECKPOINT_DIR
        / (
            f"{dataset_name.lower()}_"
            f"{model_name.lower()}_"
            f"{n_qubits}qubit_"
            f"4layer_preproc{PREPROCESSING_VERSION}.pth"
        )
    )


def save_checkpoint(
    model,
    path,
    dataset_name,
    epoch,
    best_val_loss,
):

    checkpoint = {
        "model_type":
            str(
                model.model_type
            ),

        "dataset_name":
            str(
                dataset_name
            ),

        "n_qubits":
            int(
                model.n_qubits
            ),

        "n_layers":
            4,

        "n_classes":
            int(
                model.n_classes
            ),

        "device_name":
            str(
                model.device_name
            ),

        "preprocessing_version": PREPROCESSING_VERSION,

        "state_dict":
            model.state_dict(),

        "epoch":
            int(
                epoch
            ),

        "best_val_loss":
            float(
                best_val_loss
            ),

        "torch_version":
            str(
                torch.__version__
            ),

        "pennylane_version":
            str(
                qml.__version__
            ),
    }

    torch.save(
        checkpoint,
        path,
    )

    print(
        f"Saved checkpoint: "
        f"{path}"
    )


def load_checkpoint(
    path,
):

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError(f"Checkpoint {path} uses incompatible preprocessing")

    model = (
        Quantum4LayerClassifier(
            model_type=str(
                checkpoint[
                    "model_type"
                ]
            ),

            n_qubits=int(
                checkpoint[
                    "n_qubits"
                ]
            ),

            n_classes=int(
                checkpoint.get(
                    "n_classes",
                    N_CLASSES,
                )
            ),

            device_name=str(
                checkpoint.get(
                    "device_name",
                    PENNYLANE_DEVICE,
                )
            ),
        )
    )

    model.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )

    model.eval()

    print(
        f"Loaded checkpoint: "
        f"{path}"
    )

    return model


# ============================================================
# 13. TRAINING DATASET
# ============================================================

class QuantumDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        base_dataset,
        indices: Iterable[int],
        n_qubits,
        mean,
        std,
    ):
        self.base_dataset = (
            base_dataset
        )

        self.indices = list(
            indices
        )

        self.n_qubits = (
            n_qubits
        )

        self.mean = mean
        self.std = std

    def __len__(
        self,
    ):
        return len(
            self.indices
        )

    def __getitem__(
        self,
        idx,
    ):

        image, label = (
            self.base_dataset[
                self.indices[
                    idx
                ]
            ]
        )

        x = preprocess_image(
            image,
            self.n_qubits,
            self.mean,
            self.std,
        )

        return (
            x,
            int(label),
        )


def make_train_val_loaders(
    train_dataset,
    n_qubits,
    mean,
    std,
):

    total = min(
        TRAIN_SAMPLES,
        len(
            train_dataset
        ),
    )

    if total < 2:
        raise ValueError("TRAIN_SAMPLES must be >= 2 for disjoint train/validation splits")

    rng = (
        np.random
        .default_rng(
            SEED
        )
    )

    indices = rng.permutation(
        len(
            train_dataset
        )
    )[:total]

    n_val = max(
        1,
        int(
            0.15
            * total
        ),
    )

    val_idx = (
        indices[
            :n_val
        ]
    )

    train_idx = (
        indices[
            n_val:
        ]
    )

    train_q = QuantumDataset(
        train_dataset,
        train_idx,
        n_qubits,
        mean,
        std,
    )

    val_q = QuantumDataset(
        train_dataset,
        val_idx,
        n_qubits,
        mean,
        std,
    )

    train_loader = DataLoader(
        train_q,
        batch_size=(
            TRAIN_BATCH_SIZE
        ),
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_q,
        batch_size=(
            TRAIN_BATCH_SIZE
        ),
        shuffle=False,
        num_workers=0,
    )

    return (
        train_loader,
        val_loader,
    )


# ============================================================
# 14. TRAIN / VALIDATE
# ============================================================

@torch.no_grad()
def evaluate_loss(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:

        logits = model(
            x
        )

        loss = criterion(
            logits,
            y,
        )

        total_loss += (
            float(
                loss.item()
            )
            * y.size(0)
        )

        total_correct += int(
            (
                logits.argmax(
                    dim=1
                )
                == y
            )
            .sum()
            .item()
        )

        total_count += (
            y.size(0)
        )

    mean_loss = (
        total_loss
        / max(
            1,
            total_count,
        )
    )

    accuracy = (
        total_correct
        / max(
            1,
            total_count,
        )
    )

    return (
        mean_loss,
        accuracy,
    )


def train_model(
    model,
    train_loader,
    val_loader,
    checkpoint_path,
    dataset_name,
):

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )
    )

    best_val_loss = float(
        "inf"
    )

    for epoch in range(
        1,
        TRAIN_EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0
        total_correct = 0
        total_count = 0

        progress = tqdm(
            train_loader,
            desc=(
                f"{dataset_name} / "
                f"{model.model_type} / "
                f"epoch "
                f"{epoch}/"
                f"{TRAIN_EPOCHS}"
            ),
            unit="batch",
            dynamic_ncols=True,
        )

        for x, y in progress:

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                x
            )

            loss = criterion(
                logits,
                y,
            )

            loss.backward()
            optimizer.step()

            running_loss += (
                float(
                    loss.item()
                )
                * y.size(0)
            )

            total_correct += int(
                (
                    logits.argmax(
                        dim=1
                    )
                    == y
                )
                .sum()
                .item()
            )

            total_count += (
                y.size(0)
            )

            progress.set_postfix(
                loss=round(
                    running_loss
                    / max(
                        1,
                        total_count,
                    ),
                    4,
                ),

                acc=round(
                    total_correct
                    / max(
                        1,
                        total_count,
                    ),
                    4,
                ),
            )

        val_loss, val_acc = (
            evaluate_loss(
                model,
                val_loader,
                criterion,
            )
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss="
            f"{running_loss/max(1,total_count):.4f}, "
            f"train_acc="
            f"{total_correct/max(1,total_count):.4f}, "
            f"val_loss="
            f"{val_loss:.4f}, "
            f"val_acc="
            f"{val_acc:.4f}"
        )

        if (
            val_loss
            < best_val_loss
        ):

            best_val_loss = (
                val_loss
            )

            save_checkpoint(
                model,
                checkpoint_path,
                dataset_name,
                epoch,
                best_val_loss,
            )

    return load_checkpoint(
        checkpoint_path
    )


def get_or_train_model(
    model_name,
    dataset_name,
    train_dataset,
    mean,
    std,
    n_classes,
):

    n_qubits = (
        DNN_QUBITS
        if model_name
        == "QuantumDNN4"
        else CNN_QUBITS
    )

    checkpoint_path = (
        get_checkpoint_path(
            model_name,
            dataset_name,
            n_qubits,
        )
    )

    if (
        checkpoint_path.exists()
        and not FORCE_RETRAIN
    ):

        model = (
            load_checkpoint(
                checkpoint_path
            )
        )

        if model.n_classes != n_classes or model.model_type != model_name:
            raise ValueError(
                f"Incompatible checkpoint {checkpoint_path}: "
                f"{model.n_classes} classes / {model.model_type}; expected {n_classes} / {model_name}. "
                "Set FORCE_RETRAIN=1 to replace it."
            )
        return model, checkpoint_path

    model = (
        Quantum4LayerClassifier(
            model_type=(
                model_name
            ),

            n_qubits=(
                n_qubits
            ),

            n_classes=(
                n_classes
            ),

            device_name=(
                PENNYLANE_DEVICE
            ),
        )
    )

    print(
        f"Training {model_name}: "
        f"qubits={n_qubits}, "
        f"layers=4, "
        f"parameters="
        f"{model.parameter_count:,}"
    )

    (
        train_loader,
        val_loader,
    ) = make_train_val_loaders(
        train_dataset,
        n_qubits,
        mean,
        std,
    )

    model = train_model(
        model,
        train_loader,
        val_loader,
        checkpoint_path,
        dataset_name,
    )

    return (
        model,
        checkpoint_path,
    )


# ============================================================
# 15. PREDICTION QUALITY
# ============================================================

def get_prediction_quality(
    logits,
):

    probs = (
        F.softmax(
            logits.float(),
            dim=-1,
        )
        .squeeze()
    )

    confidence = float(
        probs.max().item()
    )

    top2 = (
        torch.topk(
            logits
            .float()
            .squeeze(),
            k=2,
        )
        .values
    )

    margin = float(
        (
            top2[0]
            - top2[1]
        ).item()
    )

    entropy = float(
        -(
            probs
            * torch.log(
                probs
                + 1e-12
            )
        )
        .sum()
        .item()
    )

    return (
        round(
            confidence,
            6,
        ),

        round(
            margin,
            6,
        ),

        round(
            entropy,
            6,
        ),
    )


# ============================================================
# 16. ENERGY TRACKING
# ============================================================

def extract_energy_data(
    tracker,
    emissions_value,
):

    final_data = getattr(
        tracker,
        "final_emissions_data",
        None,
    )

    cpu_energy = (
        getattr(
            final_data,
            "cpu_energy",
            0,
        )
        if final_data
        else 0
    )

    gpu_energy = (
        getattr(
            final_data,
            "gpu_energy",
            0,
        )
        if final_data
        else 0
    )

    ram_energy = (
        getattr(
            final_data,
            "ram_energy",
            0,
        )
        if final_data
        else 0
    )

    total_energy = (
        getattr(
            final_data,
            "energy_consumed",
            0,
        )
        if final_data
        else 0
    )

    carbon_intensity = None

    if (
        emissions_value
        and total_energy
        and total_energy > 0
    ):
        carbon_intensity = (
            emissions_value
            / total_energy
        )

    return (
        float(
            cpu_energy
            or 0
        ),

        float(
            gpu_energy
            or 0
        ),

        float(
            ram_energy
            or 0
        ),

        float(
            total_energy
            or 0
        ),

        carbon_intensity,
    )


def run_with_energy_tracking(
    inference_fn,
    *args,
    project_name,
    codecarbon_filename,
    **kwargs,
):

    energy_dir = (
        OUTPUT_ROOT
        / "codecarbon"
    )

    energy_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracker = None

    if CODECARBON_AVAILABLE:

        tracker = (
            EmissionsTracker(
                project_name=
                    project_name,

                output_dir=
                    str(
                        energy_dir
                    ),

                output_file=
                    codecarbon_filename,

                log_level=
                    "error",

                save_to_file=
                    True,
            )
        )

        tracker.start()

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    t0 = (
        time.perf_counter()
    )

    result = (
        inference_fn(
            *args,
            **kwargs,
        )
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    exec_time = (
        time.perf_counter()
        - t0
    )

    cpu_energy = 0.0
    gpu_energy = 0.0
    ram_energy = 0.0
    total_energy = 0.0
    emissions_value = 0.0
    carbon_intensity = None

    if tracker is not None:

        emissions_value = (
            tracker.stop()
            or 0.0
        )

        (
            cpu_energy,
            gpu_energy,
            ram_energy,
            total_energy,
            carbon_intensity,
        ) = extract_energy_data(
            tracker,
            emissions_value,
        )

    gpu_metrics = (
        get_gpu_metrics()
    )

    return (
        result,
        exec_time,
        cpu_energy,
        gpu_energy,
        ram_energy,
        total_energy,
        emissions_value,
        carbon_intensity,
        gpu_metrics,
    )


# ============================================================
# 17. QUANTUM INFERENCE
# ============================================================

def run_quantum_inference(
    model,
    image,
    mean,
    std,
):

    x = preprocess_image(
        image,
        model.n_qubits,
        mean,
        std,
    )

    with torch.no_grad():

        logits = (
            model(
                x
            )
        )

        prediction = int(
            torch.argmax(
                logits,
                dim=1,
            ).item()
        )

    return (
        prediction,
        logits.detach(),
    )


# ============================================================
# 18. FLOPS
# ============================================================

def compute_model_flops(
    model,
    device,
):

    # Standard fvcore FLOPs are not meaningful for PennyLane
    # quantum operations, so leave as None rather than reporting
    # a misleading classical FLOP count.
    return None


# ============================================================
# 19. SAME TELEMETRY ROW AS UPLOADED CODE
#     EXCEPT quantum_computing=True
# ============================================================

def build_row(
    *,
    model_name,
    prediction,
    exec_time,
    parameters,
    true_label,
    sample_index,
    logits,
    cpu_energy,
    gpu_energy,
    ram_energy,
    total_energy,
    emissions_value,
    carbon_intensity,
    gpu_metrics,
    model_flops,
    dataset_name,
    checkpoint_path,
    n_qubits,
    fidelity=None,
    input_pixels=784,
):

    (
        confidence_score,
        logit_margin,
        entropy,
    ) = get_prediction_quality(
        logits
    )

    total_energy = (
        total_energy
        or 0.0
    )

    cpu_energy = (
        cpu_energy
        or 0.0
    )

    gpu_energy = (
        gpu_energy
        or 0.0
    )

    ram_energy = (
        ram_energy
        or 0.0
    )

    # Same proxy as uploaded code.
    input_tokens = int(input_pixels)
    output_tokens = 1

    total_tokens = (
        input_tokens
        + output_tokens
    )

    joules_per_token = 0.0
    energy_per_token_kwh = 0.0
    watts_estimated = 0.0
    gpu_energy_pct = 0.0
    cpu_energy_pct = 0.0

    if (
        total_energy > 0
        and total_tokens > 0
    ):

        energy_per_token_kwh = round(
            total_energy
            / total_tokens,
            12,
        )

        joules_per_token = round(
            (
                total_energy
                * 3_600_000
            )
            / total_tokens,
            6,
        )

        if exec_time > 0:

            watts_estimated = round(
                (
                    total_energy
                    * 3_600_000
                )
                / exec_time,
                4,
            )

        gpu_energy_pct = round(
            (
                gpu_energy
                / total_energy
            )
            * 100,
            2,
        )

        cpu_energy_pct = round(
            (
                cpu_energy
                / total_energy
            )
            * 100,
            2,
        )

    correct = (
        int(prediction)
        == int(true_label)
    )

    gpu_metrics = (
        gpu_metrics
        or {}
    )

    row = {
        # --- Identity ---
        "timestamp":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "unique_device_id":
            DEVICE_UUID,

        "device_short_id":
            DEVICE_SHORT,

        "pc_name":
            get_hostname(),

        "collection_mode":
            "automated_edge",

        # --- Sample ---
        "sample_index":
            sample_index,

        "sample_id":
            sample_index,

        "true_label":
            int(true_label),

        "prediction":
            int(prediction),

        "correct":
            correct,

        # --- Model identity ---
        "dataset":
            dataset_name,

        "model_type":
            model_name,

        "parameters":
            parameters,

        "model_flops":
            model_flops,

        "checkpoint_path":
            str(
                checkpoint_path
            ),

        # --- Prediction quality ---
        "confidence_score":
            confidence_score,

        "logit_margin":
            logit_margin,

        "entropy":
            entropy,

        # --- Timing ---
        "execution_time_sec":
            round(
                exec_time,
                10,
            ),

        # --- CodeCarbon energy ---
        "cpu_energy_kwh":
            cpu_energy,

        "gpu_energy_kwh":
            gpu_energy,

        "ram_energy_kwh":
            ram_energy,

        "total_energy_kwh":
            total_energy,

        "total_emissions_kg":
            emissions_value,

        "carbon_intensity_kgco2_kwh":
            carbon_intensity,

        "codecarbon_version":
            CODECARBON_VERSION,

        # --- Efficiency derived ---
        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "total_tokens":
            total_tokens,

        "tokens_per_second":
            (
                round(
                    total_tokens
                    / exec_time,
                    4,
                )
                if exec_time > 0
                else None
            ),

        "joules_per_token":
            joules_per_token,

        "energy_per_token_kwh":
            energy_per_token_kwh,

        "watts_estimated":
            watts_estimated,

        "gpu_energy_pct_of_total":
            gpu_energy_pct,

        "cpu_energy_pct_of_total":
            cpu_energy_pct,

        # --- CPU hardware ---
        "cpu_model":
            CPU_MODEL_NAME,

        "cpu_architecture":
            CPU_ARCH,

        "cpu_core_count":
            CPU_CORE_COUNT,

        "cpu_thread_count":
            CPU_THREAD_COUNT,

        "cpu_core":
            CPU_CORE_COUNT,

        "cpu_thread":
            CPU_THREAD_COUNT,

        "cpu_tdp_w":
            CPU_TDP_W,

        "cpu_usage_pct":
            get_cpu_usage(),

        "cpu_clock_mhz":
            get_cpu_freq(),

        "cpu_temp_c":
            get_cpu_temp(),

        "cpu_power_draw_w":
            get_cpu_power_draw_w(),

        "cpu_cores_used":
            get_cpu_cores_used(),

        # --- GPU hardware ---
        "gpu_model":
            get_gpu_name(),

        "gpu_core":
            GPU_CORE_COUNT,

        "gpu_thread":
            GPU_THREAD_COUNT,

        "gpu_driver_version":
            GPU_STATIC[
                "gpu_driver_version"
            ],

        "gpu_compute_capability":
            GPU_STATIC[
                "gpu_compute_capability"
            ],

        "gpu_power_limit_w":
            GPU_STATIC[
                "gpu_power_limit_w"
            ],

        "gpu_memory_total_mb":
            GPU_STATIC[
                "gpu_memory_total_mb"
            ],

        "gpu_power_draw_w":
            gpu_metrics.get(
                "gpu_power_draw_w"
            ),

        "gpu_utilization_pct":
            gpu_metrics.get(
                "gpu_utilization_pct"
            ),

        "gpu_temp_c":
            gpu_metrics.get(
                "gpu_temp_c"
            ),

        "gpu_memory_used_mb":
            gpu_metrics.get(
                "gpu_memory_used_mb"
            ),

        "gpu_sm_clock_mhz":
            gpu_metrics.get(
                "gpu_sm_clock_mhz"
            ),

        "gpu_memory_clock_mhz":
            gpu_metrics.get(
                "gpu_memory_clock_mhz"
            ),

        "cuda_driver_version":
            CUDA_DRIVER_VERSION,

        "cuda_available":
            torch.cuda.is_available(),

        "device_type":
            str(
                DEVICE
            ),

        # --- RAM / memory ---
        "ram_usage_pct":
            get_ram_usage(),

        "memory_footprint_mb":
            get_memory_footprint_mb(),

        "system_ram_total_gb":
            SYSTEM_RAM_TOTAL_GB,

        # --- Environment ---
        "os_name":
            OS_NAME,

        "os_version":
            OS_VERSION,

        "os_architecture":
            OS_ARCHITECTURE,

        "os_full_name":
            OS_FULL_NAME,

        "python_version":
            PYTHON_VERSION,

        "torch_version":
            TORCH_VERSION,

        # --- Model metrics ---
        "model_accuracy":
            None,

        "model_precision_weighted":
            None,

        "model_recall_weighted":
            None,

        "model_f1_weighted":
            None,

        # --- IMPORTANT CHANGE ---
        "quantum_computing":
            True,

        "model_under_attack":
            0,

        # --- Quantum fields ---
        "source_qubits":
            int(
                n_qubits
            ),

        "n_qubits":
            int(
                n_qubits
            ),

        "fidelity":
            fidelity,

        "pennylane_device":
            PENNYLANE_DEVICE,

        "pennylane_version":
            PENNYLANE_VERSION,
    }

    # Guarantee identical schema/order.
    assert (
        list(
            row.keys()
        )
        == REFERENCE_COLUMNS
    ), (
        "Telemetry schema does not "
        "match uploaded reference columns."
    )

    return row


# ============================================================
# 20. CSV HELPERS
# ============================================================

def get_output_path(
    model_name,
    dataset_name,
):

    return (
        DEVICE_LOG_DIR
        / (
            f"quantum_dnn_cnn_dataset_"
            f"{DEVICE_SHORT}_"
            f"{dataset_name.lower()}_"
            f"{model_name.lower()}_preproc{PREPROCESSING_VERSION}.csv"
        )
    )


def append_rows(
    rows,
    file_path,
):

    if not rows:
        return

    new_df = (
        pd.DataFrame(
            rows
        )
        .reindex(
            columns=
                REFERENCE_COLUMNS
        )
    )

    if file_path.exists():

        try:
            old_df = (
                pd.read_csv(
                    file_path,
                    on_bad_lines=
                        "skip",
                )
            )

            old_df = (
                old_df.reindex(
                    columns=
                        REFERENCE_COLUMNS
                )
            )

            pd.concat(
                [
                    old_df,
                    new_df,
                ],
                ignore_index=True,
            ).to_csv(
                file_path,
                index=False,
            )

            return

        except Exception:
            pass

    new_df.to_csv(
        file_path,
        index=False,
    )


def get_existing_count(
    file_path,
    model_name,
):

    if not file_path.exists():
        return 0

    try:
        df = (
            pd.read_csv(
                file_path,
                on_bad_lines=
                    "skip",
            )
        )

        if (
            "model_type"
            not in df.columns
        ):
            return 0

        return int(
            (
                df[
                    "model_type"
                ]
                .astype(str)
                .str.strip()
                == model_name
            )
            .sum()
        )

    except Exception:
        return 0


def backfill_model_metrics(
    file_path,
    model_name,
):

    if not file_path.exists():
        return None

    df = pd.read_csv(
        file_path,
        on_bad_lines=
            "skip",
    )

    if (
        "model_type"
        not in df.columns
    ):
        return None

    mask = (
        df[
            "model_type"
        ]
        .astype(str)
        .str.strip()
        == model_name
    )

    model_df = (
        df.loc[
            mask
        ]
        .dropna(
            subset=[
                "true_label",
                "prediction",
            ]
        )
    )

    if model_df.empty:
        return None

    y_true = (
        model_df[
            "true_label"
        ]
        .astype(int)
        .tolist()
    )

    y_pred = (
        model_df[
            "prediction"
        ]
        .astype(int)
        .tolist()
    )

    accuracy = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    precision = float(
        precision_score(
            y_true,
            y_pred,
            average=
                "weighted",
            zero_division=
                0,
        )
    )

    recall = float(
        recall_score(
            y_true,
            y_pred,
            average=
                "weighted",
            zero_division=
                0,
        )
    )

    f1 = float(
        f1_score(
            y_true,
            y_pred,
            average=
                "weighted",
            zero_division=
                0,
        )
    )

    df.loc[
        mask,
        "model_accuracy",
    ] = accuracy

    df.loc[
        mask,
        "model_precision_weighted",
    ] = precision

    df.loc[
        mask,
        "model_recall_weighted",
    ] = recall

    df.loc[
        mask,
        "model_f1_weighted",
    ] = f1

    df = df.reindex(
        columns=
            REFERENCE_COLUMNS
    )

    df.to_csv(
        file_path,
        index=False,
    )

    print(
        f"{model_name} metrics -> "
        f"Accuracy: "
        f"{accuracy:.4f}, "
        f"Precision: "
        f"{precision:.4f}, "
        f"Recall: "
        f"{recall:.4f}, "
        f"F1: "
        f"{f1:.4f}"
    )

    return {
        "accuracy":
            accuracy,

        "precision_weighted":
            precision,

        "recall_weighted":
            recall,

        "f1_weighted":
            f1,
    }


# ============================================================
# 21. TELEMETRY COLLECTION
# ============================================================

def collect_for_model(
    base_dataset,
    model,
    checkpoint_path,
    model_name,
    dataset_name,
    mean,
    std,
    num_samples,
    flush_every,
    input_pixels=784,
):

    print(
        f"\nCollecting "
        f"{num_samples} samples "
        f"for {dataset_name} / "
        f"{model_name}"
    )

    print(
        f"Device UUID : "
        f"{DEVICE_UUID}"
    )

    print(
        f"Device short: "
        f"{DEVICE_SHORT}"
    )

    output_path = (
        get_output_path(
            model_name,
            dataset_name,
        )
    )

    parameters = (
        model.parameter_count
    )

    model_flops = (
        compute_model_flops(
            model,
            DEVICE,
        )
    )

    limit = min(
        num_samples,
        len(
            base_dataset
        ),
    )

    existing_count = (
        get_existing_count(
            output_path,
            model_name,
        )
    )

    if (
        existing_count
        >= limit
    ):

        print(
            f"{model_name}: "
            f"already complete "
            f"({existing_count}/"
            f"{limit})"
        )

        backfill_model_metrics(
            output_path,
            model_name,
        )

        return

    rows = []

    print(
        f"{model_name}: "
        f"resuming from "
        f"{existing_count}/"
        f"{limit}"
    )

    progress_bar = tqdm(
        range(
            existing_count,
            limit,
        ),
        desc=(
            f"{model_name}"
        ),
        unit="sample",
        dynamic_ncols=True,
    )

    model.eval()

    for i in progress_bar:

        image_tensor, true_label = (
            base_dataset[
                i
            ]
        )

        (
            (
                prediction,
                logits,
            ),
            exec_time,
            cpu_energy,
            gpu_energy,
            ram_energy,
            total_energy,
            emissions_value,
            carbon_intensity,
            gpu_metrics,
        ) = run_with_energy_tracking(
            run_quantum_inference,

            model,
            image_tensor,

            mean=mean,
            std=std,

            project_name=(
                f"{model_name}_"
                f"{dataset_name}_"
                f"quantum_test"
            ),

            codecarbon_filename=(
                f"codecarbon_"
                f"{model_name.lower()}_"
                f"{dataset_name.lower()}.csv"
            ),
        )

        row = build_row(
            model_name=
                model_name,

            prediction=
                prediction,

            exec_time=
                exec_time,

            parameters=
                parameters,

            true_label=
                int(
                    true_label
                ),

            sample_index=
                i,

            logits=
                logits,

            cpu_energy=
                cpu_energy,

            gpu_energy=
                gpu_energy,

            ram_energy=
                ram_energy,

            total_energy=
                total_energy,

            emissions_value=
                emissions_value,

            carbon_intensity=
                carbon_intensity,

            gpu_metrics=
                gpu_metrics,

            model_flops=
                model_flops,

            dataset_name=
                dataset_name,

            checkpoint_path=
                checkpoint_path,

            n_qubits=
                model.n_qubits,

            fidelity=
                None,
            input_pixels=input_pixels,
        )

        rows.append(
            row
        )

        progress_bar.set_postfix({
            "pred":
                prediction,

            "true":
                int(
                    true_label
                ),

            "time_s":
                round(
                    exec_time,
                    3,
                ),

            "done":
                i + 1,
        })

        if (
            (i + 1)
            % flush_every
            == 0
        ):

            append_rows(
                rows,
                output_path,
            )

            rows = []

    if rows:
        append_rows(
            rows,
            output_path,
        )

    backfill_model_metrics(
        output_path,
        model_name,
    )

    print(
        f"{model_name}: "
        f"finished "
        f"{limit}/{limit} "
        f"-> "
        f"{output_path}"
    )


# ============================================================
# 22. MAIN
# ============================================================

def main():
    selected = [name.strip() for name in os.getenv(
        "DATASETS", "MNIST,FashionMNIST,KMNIST,CIFAR100,TinyImageNet200"
    ).split(",") if name.strip()]
    invalid = [x for x in selected if x not in DATASET_CONFIGS]
    if invalid:
        raise ValueError(f"Unknown datasets: {invalid}. Available: {list(DATASET_CONFIGS)}")

    print("=" * 72)
    print("FOUR-LAYER QUANTUM CNN/DNN SIMULATOR TELEMETRY")
    print("Datasets:", selected)
    print(f"Qubits: CNN={CNN_QUBITS}, DNN={DNN_QUBITS}; layers={N_LAYERS}")
    print(f"Train samples={TRAIN_SAMPLES}, epochs={TRAIN_EPOCHS}; test samples={NUM_TEST_SAMPLES}")
    print("Device:", DEVICE, " PennyLane:", PENNYLANE_DEVICE)
    print("Preprocessing version:", PREPROCESSING_VERSION)

    if DEVICE.type != "cpu" and PENNYLANE_DEVICE == "default.qubit":
        raise RuntimeError(
            "default.qubit runs on CPU. Set DEVICE_MODE=cpu for consistent "
            "PyTorch/PennyLane tensors and valid CPU-side telemetry."
        )

    for dataset_name in selected:
        config = DATASET_CONFIGS[dataset_name]
        print(f"\n{'=' * 72}\nDataset: {dataset_name} ({config['num_classes']} classes)\n{'=' * 72}")
        train_dataset, test_dataset = load_dataset_pair(dataset_name, config)

        for model_name in QUANTUM_MODELS:
            model, checkpoint_path = get_or_train_model(
                model_name=model_name,
                dataset_name=dataset_name,
                train_dataset=train_dataset,
                mean=config["mean"],
                std=config["std"],
                n_classes=config["num_classes"],
            )
            collect_for_model(
                base_dataset=test_dataset,
                model=model,
                checkpoint_path=checkpoint_path,
                model_name=model_name,
                dataset_name=dataset_name,
                mean=config["mean"],
                std=config["std"],
                num_samples=NUM_TEST_SAMPLES,
                flush_every=FLUSH_EVERY,
                input_pixels=config["channels"] * config["image_size"] ** 2,
            )
            del model
    print("\nDone.")


if __name__ == "__main__":
    main()
