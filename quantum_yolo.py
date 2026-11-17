# ============================================================

# 1-QUBIT HYBRID YOLO CLASSIFIER

# MNIST / FashionMNIST / KMNIST

#

# Behavior:

#   1. For each dataset, look for a saved hybrid YOLO checkpoint.

#   2. If the checkpoint exists, load it.

#   3. If it does not exist, build YOLO26n-CLS with a 10-qubit

#      PennyLane classification head, train for 10 epochs, and save it.

#   4. Evaluate the saved/loaded model on the test set.

#

# IMPORTANT:

# This is a REAL hybrid quantum-classical YOLO classifier.

# It is different from the uploaded QYOLO paper, which is

# "quantum-inspired" and does not use physical/simulated qubits.

#

# Install:

#   pip install torch torchvision ultralytics pennylane \\

#               pennylane-lightning scikit-learn tqdm

#

# Run:

#   python yolo_10qubit_train_or_load.py

#

# Optional environment variables:

#   DEVICE_MODE=cpu

#   TRAIN_BATCH_SIZE=8

#   TEST_BATCH_SIZE=16

#   TRAIN_SAMPLES=0        # 0 = use all training samples

#   TEST_SAMPLES=10000     # 0 = use all test samples

#   FORCE_RETRAIN=0

#   PENNYLANE_DEVICE=default.qubit

# ============================================================



from __future__ import annotations



import os
import hashlib
import platform
import socket
import sys
import time

import random

from pathlib import Path



import numpy as np

import pennylane as qml

import torch

import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import psutil

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

)

from torch.utils.data import DataLoader, Subset

from torchvision import datasets, transforms

from tqdm.auto import tqdm

from ultralytics import YOLO





# ============================================================

# CONFIGURATION

# ============================================================



BASE_DIR = Path("../classical_data")

CHECKPOINT_DIR = Path("./quantum_checkpoints")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)



SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)



EPOCHS = int(

    os.getenv("TRAIN_EPOCHS", "10")

)

IMG_SIZE = 64



N_QUBITS = 1

N_Q_LAYERS = 4

N_CLASSES = 10



LEARNING_RATE = 1e-3

WEIGHT_DECAY = 5e-4



TRAIN_BATCH_SIZE = int(

    os.getenv("TRAIN_BATCH_SIZE", "16")

)



TEST_BATCH_SIZE = int(

    os.getenv("TEST_BATCH_SIZE", "16")

)



# 0 means use the full dataset.

TRAIN_SAMPLES = int(

    os.getenv("TRAIN_SAMPLES", "0")

)



TEST_SAMPLES = int(

    os.getenv("NUM_TEST_SAMPLES", "10")

)



FORCE_RETRAIN = (

    os.getenv("FORCE_RETRAIN", "0").strip() == "1"

)



PENNYLANE_DEVICE = os.getenv(

    "PENNYLANE_DEVICE",

    "default.qubit",

)



DEVICE_MODE = os.getenv(

    "DEVICE_MODE",

    "cpu",

).lower()



DEVICE = torch.device("cpu")





DATASET_CONFIGS = {

    "MNIST": {

        "dataset_class": datasets.MNIST,

        "key": "mnist",

    },

    "FashionMNIST": {

        "dataset_class": datasets.FashionMNIST,

        "key": "fashionmnist",

    },

    "KMNIST": {

        "dataset_class": datasets.KMNIST,

        "key": "kmnist",

    },

}





# ============================================================

# IMAGE PREPROCESSING

# ============================================================



# YOLO classification expects a 3-channel image.

# MNIST-family datasets are grayscale, so replicate to RGB.

IMAGE_TRANSFORM = transforms.Compose([

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.Grayscale(num_output_channels=3),

    transforms.ToTensor(),

])





# ============================================================

# 1-QUBIT QUANTUM CLASSIFICATION HEAD

# ============================================================



def build_quantum_circuit(

    n_qubits: int = N_QUBITS,

    n_layers: int = N_Q_LAYERS,

    device_name: str = PENNYLANE_DEVICE,

):

    """

    Variational quantum circuit.



    Input:

        x       -> n_qubits encoded angles

        weights -> trainable rotations



    Output:

        one Pauli-Z expectation value per qubit



    Since N_QUBITS = 1 and MNIST-family datasets have 10 classes,

    the ten expectation values are used as ten class features.

    """



    dev = qml.device(

        device_name,

        wires=n_qubits,

    )



    @qml.qnode(

        dev,

        interface="torch",

        diff_method="best",

    )

    def circuit(x, weights):



        # -------------------------------

        # Angle encoding

        # -------------------------------

        for q in range(n_qubits):

            qml.RY(

                x[q],

                wires=q,

            )



            qml.RZ(

                x[q],

                wires=q,

            )



        # -------------------------------

        # Trainable variational layers

        # -------------------------------

        for layer in range(n_layers):



            for q in range(n_qubits):

                qml.Rot(

                    weights[layer, q, 0],

                    weights[layer, q, 1],

                    weights[layer, q, 2],

                    wires=q,

                )



            # Ring entanglement

            if n_qubits > 1:

                for q in range(n_qubits):

                    qml.CNOT(

                        wires=[

                            q,

                            (q + 1) % n_qubits,

                        ]

                    )



        # 1 expectation value

        return [

            qml.expval(

                qml.PauliZ(q)

            )

            for q in range(n_qubits)

        ]



    return circuit





class QuantumYOLOHead(nn.Module):

    """

    Replaces YOLO's final classical Linear classifier.



    Pipeline:



        YOLO feature vector

              |

              v

        Linear(in_features -> 1)

              |

            tanh*pi

              |

              v

         1-qubit VQC

              |

              v

        1 expectation value

              |

              v

        Linear(1 -> 10)

              |

              v

        10 class logits

    """



    def __init__(

        self,

        in_features: int,

        n_qubits: int = N_QUBITS,

        n_layers: int = N_Q_LAYERS,

    ):

        super().__init__()



        self.in_features = int(in_features)

        self.n_qubits = int(n_qubits)

        self.n_layers = int(n_layers)



        # Compress YOLO's high-dimensional feature vector

        # to one angle per qubit.

        self.feature_to_qubits = nn.Linear(

            self.in_features,

            self.n_qubits,

        )



        # Trainable VQC parameters.

        self.q_weights = nn.Parameter(

            0.01 * torch.randn(

                self.n_layers,

                self.n_qubits,

                3,

            )

        )



        self.qnode = build_quantum_circuit(

            n_qubits=self.n_qubits,

            n_layers=self.n_layers,

            device_name=PENNYLANE_DEVICE,

        )



        # IMPORTANT: the circuit returns n_qubits quantum features,

        # but MNIST/FashionMNIST/KMNIST always have 10 classes.

        # This layer maps 1 quantum feature -> 10 class logits when

        # N_QUBITS = 1.

        self.classifier = nn.Linear(

            self.n_qubits,

            N_CLASSES,

        )



    def _forward_single(

        self,

        features: torch.Tensor,

    ) -> torch.Tensor:



        # Compress YOLO features to exactly 1 quantum input angle.

        angles = self.feature_to_qubits(

            features

        )



        # Bound encoding angles.

        angles = (

            torch.tanh(angles)

            * torch.pi

        )



        q_out = self.qnode(

            angles,

            self.q_weights,

        )



        # PennyLane can return a tuple/list of scalar tensors.

        if isinstance(q_out, torch.Tensor):

            q_features = q_out

        else:

            q_features = torch.stack(

                list(q_out)

            )



        q_features = q_features.to(

            dtype=torch.float32

        )



        logits = self.classifier(

            q_features

        )



        return logits



    def forward(

        self,

        x: torch.Tensor,

    ) -> torch.Tensor:



        # YOLO classification head passes shape:

        #     (batch, in_features)



        if x.ndim == 1:

            return self._forward_single(

                x

            ).unsqueeze(0)



        # PennyLane simulation is evaluated sample-by-sample.

        outputs = [

            self._forward_single(

                sample

            )

            for sample in x

        ]



        return torch.stack(

            outputs,

            dim=0,

        )





# ============================================================

# FIND / REPLACE YOLO FINAL LINEAR CLASSIFIER

# ============================================================



def find_last_linear(

    module: nn.Module,

):

    """

    Return:

        parent_module,

        child_name,

        linear_module



    for the last nn.Linear layer in the model.

    """



    last = None



    for full_name, child in module.named_modules():

        if isinstance(

            child,

            nn.Linear,

        ):

            last = (

                full_name,

                child,

            )



    if last is None:

        raise RuntimeError(

            "Could not find a final nn.Linear "

            "layer in the YOLO classification model."

        )



    full_name, linear = last



    parts = full_name.split(".")



    parent = module



    for part in parts[:-1]:

        if part.isdigit():

            parent = parent[

                int(part)

            ]

        else:

            parent = getattr(

                parent,

                part,

            )



    child_name = parts[-1]



    return (

        parent,

        child_name,

        linear,

    )





def set_child_module(

    parent: nn.Module,

    child_name: str,

    new_module: nn.Module,

):

    if child_name.isdigit():

        parent[

            int(child_name)

        ] = new_module

    else:

        setattr(

            parent,

            child_name,

            new_module,

        )





def build_quantum_yolo():

    """

    Build YOLO26n classification model and replace only its

    final classical linear classifier with the 10-qubit head.

    """



    yolo = YOLO(

        "yolo26n-cls.yaml"

    )



    torch_model = yolo.model



    (

        parent,

        child_name,

        old_linear,

    ) = find_last_linear(

        torch_model

    )



    in_features = int(

        old_linear.in_features

    )



    print(

        "Replacing YOLO final classifier:"

    )



    print(

        f"  Linear({in_features}, "

        f"{old_linear.out_features})"

    )



    print(

        f"  -> 1-qubit quantum head"

    )



    quantum_head = QuantumYOLOHead(

        in_features=in_features,

        n_qubits=N_QUBITS,

        n_layers=N_Q_LAYERS,

    )



    set_child_module(

        parent,

        child_name,

        quantum_head,

    )



    torch_model = torch_model.to(

        DEVICE

    )



    return torch_model





# ============================================================

# CHECKPOINTS

# ============================================================



def get_checkpoint_path(

    dataset_key: str,

) -> Path:



    return (

        CHECKPOINT_DIR

        / (

            f"yolo26n_{dataset_key}_"

            f"{N_QUBITS}qubit_cpu.pth"

        )

    )





def save_checkpoint(

    model: nn.Module,

    checkpoint_path: Path,

    dataset_name: str,

    epoch: int,

    best_val_accuracy: float,

):

    checkpoint = {

        "dataset_name": str(

            dataset_name

        ),



        "model_name":

            "YOLO26n-CLS-1Qubit",



        "n_qubits": int(

            N_QUBITS

        ),



        "n_q_layers": int(

            N_Q_LAYERS

        ),



        "epoch": int(

            epoch

        ),



        "best_val_accuracy": float(

            best_val_accuracy

        ),



        "pennylane_device": str(

            PENNYLANE_DEVICE

        ),



        "pennylane_version": str(

            qml.__version__

        ),



        "torch_version": str(

            torch.__version__

        ),



        "state_dict":

            model.state_dict(),

    }



    torch.save(

        checkpoint,

        checkpoint_path,

    )



    print(

        f"Saved checkpoint: "

        f"{checkpoint_path}"

    )





def load_checkpoint(

    checkpoint_path: Path,

):

    """

    Rebuild the architecture first, then load the trained weights.

    """



    model = build_quantum_yolo()



    # PyTorch 2.6+:

    # This checkpoint is generated by this script,

    # so it is a trusted local file.

    checkpoint = torch.load(

        checkpoint_path,

        map_location=DEVICE,

        weights_only=False,

    )



    saved_qubits = int(

        checkpoint.get(

            "n_qubits",

            N_QUBITS,

        )

    )



    if saved_qubits != N_QUBITS:

        raise RuntimeError(

            f"Checkpoint uses {saved_qubits} qubits, "

            f"but current code uses {N_QUBITS}."

        )



    model.load_state_dict(

        checkpoint[

            "state_dict"

        ]

    )



    model = model.to(

        DEVICE

    )



    model.eval()



    print(

        f"Loaded checkpoint: "

        f"{checkpoint_path}"

    )



    print(

        f"Saved epoch: "

        f"{checkpoint.get('epoch', '?')}"

    )



    print(

        "Saved best validation accuracy: "

        f"{checkpoint.get('best_val_accuracy', '?')}"

    )



    return model





# ============================================================

# DATASETS

# ============================================================



def make_dataset(

    dataset_class,

    train: bool,

):

    return dataset_class(

        root=str(

            BASE_DIR

        ),

        train=train,

        download=True,

        transform=IMAGE_TRANSFORM,

    )





def maybe_subset(

    dataset,

    n_samples: int,

    seed: int,

):

    if (

        n_samples <= 0

        or n_samples >= len(dataset)

    ):

        return dataset



    generator = torch.Generator()

    generator.manual_seed(

        seed

    )



    indices = torch.randperm(

        len(dataset),

        generator=generator,

    )[:n_samples]



    return Subset(

        dataset,

        indices.tolist(),

    )





def make_loaders(

    dataset_class,

):

    full_train = make_dataset(

        dataset_class,

        train=True,

    )



    test_dataset = make_dataset(

        dataset_class,

        train=False,

    )



    full_train = maybe_subset(

        full_train,

        TRAIN_SAMPLES,

        SEED,

    )



    test_dataset = maybe_subset(

        test_dataset,

        TEST_SAMPLES,

        SEED + 1,

    )



    # Use 90/10 train/validation split.

    total = len(

        full_train

    )



    n_val = max(

        1,

        int(

            0.10 * total

        ),

    )



    n_train = (

        total - n_val

    )



    train_dataset, val_dataset = (

        torch.utils.data.random_split(

            full_train,

            [

                n_train,

                n_val,

            ],

            generator=torch.Generator()

            .manual_seed(

                SEED

            ),

        )

    )



    train_loader = DataLoader(

        train_dataset,

        batch_size=TRAIN_BATCH_SIZE,

        shuffle=True,

        num_workers=0,

    )



    val_loader = DataLoader(

        val_dataset,

        batch_size=TEST_BATCH_SIZE,

        shuffle=False,

        num_workers=0,

    )



    test_loader = DataLoader(

        test_dataset,

        batch_size=TEST_BATCH_SIZE,

        shuffle=False,

        num_workers=0,

    )



    return (

        train_loader,

        val_loader,

        test_loader,

    )





# ============================================================

# OUTPUT NORMALIZATION

# ============================================================



def extract_logits(

    output,

):

    """

    Ultralytics model output can vary slightly by version.

    This helper extracts the classification logits tensor.

    """



    if isinstance(

        output,

        torch.Tensor,

    ):

        return output



    if isinstance(

        output,

        (list, tuple),

    ):

        # Prefer a B x 10 tensor.

        for value in output:

            if (

                isinstance(

                    value,

                    torch.Tensor,

                )

                and value.ndim == 2

                and value.shape[-1] == N_CLASSES

            ):

                return value



        # Fallback to first tensor.

        for value in output:

            if isinstance(

                value,

                torch.Tensor,

            ):

                return value



    if isinstance(

        output,

        dict,

    ):

        for key in (

            "logits",

            "preds",

            "output",

        ):

            value = output.get(

                key

            )



            if isinstance(

                value,

                torch.Tensor,

            ):

                return value



    raise RuntimeError(

        "Could not extract classification logits "

        "from YOLO model output."

    )





# ============================================================

# VALIDATION

# ============================================================



@torch.no_grad()

def evaluate(

    model,

    loader,

    criterion,

):

    model.eval()



    total_loss = 0.0

    total_correct = 0

    total_count = 0



    for images, labels in tqdm(

        loader,

        desc="Validation",

        leave=False,

    ):

        images = images.to(

            DEVICE

        )



        labels = labels.to(

            DEVICE

        )



        output = model(

            images

        )



        logits = extract_logits(

            output

        )



        loss = criterion(

            logits,

            labels,

        )



        total_loss += (

            float(

                loss.item()

            )

            * labels.size(0)

        )



        predictions = (

            logits.argmax(

                dim=1

            )

        )



        total_correct += int(

            (

                predictions

                == labels

            )

            .sum()

            .item()

        )



        total_count += (

            labels.size(0)

        )



    return (

        total_loss

        / max(

            1,

            total_count,

        ),



        total_correct

        / max(

            1,

            total_count,

        ),

    )





# ============================================================

# TRAINING

# ============================================================



def train_model(

    dataset_name: str,

    model: nn.Module,

    train_loader,

    val_loader,

    checkpoint_path: Path,

):

    criterion = (

        nn.CrossEntropyLoss()

    )



    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

    )



    best_val_accuracy = -1.0



    for epoch in range(

        1,

        EPOCHS + 1,

    ):

        model.train()



        total_loss = 0.0

        total_correct = 0

        total_count = 0



        progress = tqdm(

            train_loader,

            desc=(

                f"{dataset_name} "

                f"epoch {epoch}/{EPOCHS}"

            ),

            unit="batch",

        )



        for images, labels in progress:

            images = images.to(

                DEVICE

            )



            labels = labels.to(

                DEVICE

            )



            optimizer.zero_grad(

                set_to_none=True

            )



            output = model(

                images

            )



            logits = extract_logits(

                output

            )



            loss = criterion(

                logits,

                labels,

            )



            loss.backward()



            optimizer.step()



            batch_size = (

                labels.size(0)

            )



            total_loss += (

                float(

                    loss.item()

                )

                * batch_size

            )



            predictions = (

                logits.argmax(

                    dim=1

                )

            )



            total_correct += int(

                (

                    predictions

                    == labels

                )

                .sum()

                .item()

            )



            total_count += (

                batch_size

            )



            progress.set_postfix(

                loss=round(

                    total_loss

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



        train_loss = (

            total_loss

            / max(

                1,

                total_count,

            )

        )



        train_accuracy = (

            total_correct

            / max(

                1,

                total_count,

            )

        )



        (

            val_loss,

            val_accuracy,

        ) = evaluate(

            model,

            val_loader,

            criterion,

        )



        print(

            f"\nEpoch {epoch}: "

            f"train_loss={train_loss:.4f}, "

            f"train_acc={train_accuracy:.4f}, "

            f"val_loss={val_loss:.4f}, "

            f"val_acc={val_accuracy:.4f}"

        )



        # Save best checkpoint.

        if (

            val_accuracy

            > best_val_accuracy

        ):

            best_val_accuracy = (

                val_accuracy

            )



            save_checkpoint(

                model=model,

                checkpoint_path=(

                    checkpoint_path

                ),

                dataset_name=(

                    dataset_name

                ),

                epoch=epoch,

                best_val_accuracy=(

                    best_val_accuracy

                ),

            )



    # Return the best model, not just the last epoch.

    return load_checkpoint(

        checkpoint_path

    )





# ============================================================

# TRAIN IF MISSING / LOAD IF AVAILABLE

# ============================================================



def get_or_train_model(

    dataset_name: str,

    dataset_key: str,

    train_loader,

    val_loader,

):

    checkpoint_path = (

        get_checkpoint_path(

            dataset_key

        )

    )



    if (

        checkpoint_path.exists()

        and not FORCE_RETRAIN

    ):

        print(

            "\nSaved 1-qubit YOLO model found."

        )



        model = load_checkpoint(

            checkpoint_path

        )



        return (

            model,

            checkpoint_path,

        )



    print(

        "\nNo saved 1-qubit YOLO model found."

    )



    print(

        f"Training for {EPOCHS} epochs "

        f"with {N_QUBITS} qubits..."

    )



    model = build_quantum_yolo()



    model = train_model(

        dataset_name=dataset_name,

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        checkpoint_path=(

            checkpoint_path

        ),

    )



    return (

        model,

        checkpoint_path,

    )





# ============================================================
# TELEMETRY SUPPORT
# ============================================================

def get_cpu_model():
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info().get("brand_raw", "Unknown")
    except Exception:
        return platform.processor() or "Unknown"

CPU_MODEL_NAME = get_cpu_model()
CPU_ARCH = platform.machine()
CPU_TDP_W = None
PYTHON_VERSION = sys.version.split()[0]
TORCH_VERSION = torch.__version__
OS_NAME = platform.system()
OS_VERSION = platform.version()
OS_ARCHITECTURE = platform.machine()
SYSTEM_RAM_TOTAL_GB = round(psutil.virtual_memory().total / (1024 ** 3), 2)
CPU_CORE_COUNT = psutil.cpu_count(logical=False)
CPU_THREAD_COUNT = psutil.cpu_count(logical=True)


def get_os_full_name():
    system = platform.system()
    architecture = platform.machine()
    if system == "Windows":
        return f"Windows {platform.release()} {architecture}"
    if system == "Linux":
        try:
            info = {}
            with open("/etc/os-release", "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        info[k] = v.strip('"')
            return f"{info.get('PRETTY_NAME', 'Linux')} {architecture}"
        except Exception:
            return f"Linux {platform.release()} {architecture}"
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]} {architecture}"
    return f"{system} {platform.release()} {architecture}"

OS_FULL_NAME = get_os_full_name()


def make_stable_device_id():
    raw = f"{socket.gethostname()}-{platform.system()}-{platform.machine()}-{CPU_MODEL_NAME}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

DEVICE_UUID = make_stable_device_id()
DEVICE_SHORT = DEVICE_UUID[:8]
OUTPUT_ROOT = Path.cwd() / "test_results"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DEVICE_LOG_DIR = OUTPUT_ROOT / DEVICE_SHORT
DEVICE_LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_hostname():
    return socket.gethostname()

try:
    from codecarbon import EmissionsTracker
    import codecarbon
    CODECARBON_AVAILABLE = True
    CODECARBON_VERSION = codecarbon.__version__
except Exception:
    EmissionsTracker = None
    CODECARBON_AVAILABLE = False
    CODECARBON_VERSION = "unavailable"
    print("CodeCarbon unavailable. Energy values will be 0.")

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0) if torch.cuda.is_available() else None
except Exception:
    NVML_AVAILABLE = False
    NVML_HANDLE = None
    print("pynvml unavailable.")


def get_cuda_driver_version():
    if not NVML_AVAILABLE:
        return None
    try:
        v = pynvml.nvmlSystemGetDriverVersion()
        return v.decode("utf-8") if isinstance(v, bytes) else v
    except Exception:
        return None

CUDA_DRIVER_VERSION = get_cuda_driver_version()


def get_gpu_static():
    result = {
        "gpu_driver_version": CUDA_DRIVER_VERSION,
        "gpu_compute_capability": None,
        "gpu_power_limit_w": None,
        "gpu_memory_total_mb": None,
    }
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            result["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        except Exception:
            pass
    if not NVML_AVAILABLE or NVML_HANDLE is None:
        return result
    try:
        result["gpu_power_limit_w"] = round(pynvml.nvmlDeviceGetPowerManagementLimit(NVML_HANDLE) / 1000.0, 2)
        mem = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE)
        result["gpu_memory_total_mb"] = round(mem.total / (1024 ** 2), 2)
    except Exception:
        pass
    return result

GPU_STATIC = get_gpu_static()


def get_gpu_core_thread():
    if not torch.cuda.is_available():
        return None, None
    try:
        props = torch.cuda.get_device_properties(0)
        sm = props.multi_processor_count
        cpsm = {2:32,3:192,5:128,6:64,7:64,8:128,9:128}.get(props.major,64)
        return sm*cpsm, sm*props.max_threads_per_multi_processor
    except Exception:
        return None, None

GPU_CORE_COUNT, GPU_THREAD_COUNT = get_gpu_core_thread()


def get_gpu_name():
    try:
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    except Exception:
        return "Unknown"


def get_cpu_usage():
    try: return psutil.cpu_percent(interval=None)
    except Exception: return None


def get_cpu_freq():
    try:
        f = psutil.cpu_freq()
        return round(f.current, 2) if f else None
    except Exception: return None


def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if not temps: return None
        for key in ("coretemp","k10temp","cpu_thermal","acpitz"):
            if key in temps:
                vals=[x.current for x in temps[key] if x.current is not None and x.current>0]
                if vals: return round(sum(vals)/len(vals),1)
    except Exception: pass
    return None


def get_cpu_power_draw_w():
    return None


def get_cpu_cores_used():
    try: return sum(1 for x in psutil.cpu_percent(percpu=True) if x > 1.0)
    except Exception: return None


def get_memory_footprint_mb():
    try: return round(psutil.Process(os.getpid()).memory_info().rss/(1024**2),4)
    except Exception: return None


def get_gpu_metrics():
    out={"gpu_power_draw_w":None,"gpu_utilization_pct":None,"gpu_temp_c":None,"gpu_memory_used_mb":None,"gpu_sm_clock_mhz":None,"gpu_memory_clock_mhz":None}
    if not NVML_AVAILABLE or NVML_HANDLE is None: return out
    try:
        power=pynvml.nvmlDeviceGetPowerUsage(NVML_HANDLE)
        util=pynvml.nvmlDeviceGetUtilizationRates(NVML_HANDLE)
        temp=pynvml.nvmlDeviceGetTemperature(NVML_HANDLE,pynvml.NVML_TEMPERATURE_GPU)
        mem=pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE)
        sm=pynvml.nvmlDeviceGetClockInfo(NVML_HANDLE,pynvml.NVML_CLOCK_SM)
        mc=pynvml.nvmlDeviceGetClockInfo(NVML_HANDLE,pynvml.NVML_CLOCK_MEM)
        return {"gpu_power_draw_w":round(power/1000.0,2),"gpu_utilization_pct":util.gpu,"gpu_temp_c":temp,"gpu_memory_used_mb":round(mem.used/(1024**2),2),"gpu_sm_clock_mhz":sm,"gpu_memory_clock_mhz":mc}
    except Exception: return out


def get_prediction_quality(logits):
    probs = F.softmax(logits.float(), dim=-1).squeeze()
    confidence_score = float(probs.max().item())
    top2 = torch.topk(logits.float().squeeze(), k=2).values
    logit_margin = float((top2[0]-top2[1]).item())
    entropy = float(-(probs*torch.log(probs+1e-12)).sum().item())
    return round(confidence_score,6), round(logit_margin,6), round(entropy,6)


def run_single_inference_with_telemetry(model, image, dataset_key):
    energy_dir = OUTPUT_ROOT / "codecarbon"
    energy_dir.mkdir(parents=True, exist_ok=True)
    tracker = None
    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name=f"yolo_1qubit_{dataset_key}_test",
            output_dir=str(energy_dir),
            output_file=f"codecarbon_yolo_1qubit_{dataset_key}.csv",
            log_level="error",
            save_to_file=True,
        )
        tracker.start()
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t0=time.perf_counter()
    output=model(image)
    logits=extract_logits(output)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    exec_time=time.perf_counter()-t0
    cpu_energy=gpu_energy=ram_energy=total_energy=0.0
    emissions_value=0.0
    carbon_intensity=None
    if tracker is not None:
        emissions_value=tracker.stop()
        fd=getattr(tracker,"final_emissions_data",None)
        cpu_energy=float(getattr(fd,"cpu_energy",0) or 0)
        gpu_energy=float(getattr(fd,"gpu_energy",0) or 0)
        ram_energy=float(getattr(fd,"ram_energy",0) or 0)
        total_energy=float(getattr(fd,"energy_consumed",0) or 0)
        if emissions_value is not None and total_energy>0:
            carbon_intensity=float(emissions_value)/total_energy
    return logits,exec_time,cpu_energy,gpu_energy,ram_energy,total_energy,emissions_value,carbon_intensity,get_gpu_metrics()


def build_telemetry_row(model_name,prediction,exec_time,parameters,true_label,sample_index,logits=None,cpu_energy=0,gpu_energy=0,ram_energy=0,total_energy=0,emissions_value=0,carbon_intensity=None,gpu_metrics=None,model_flops=None,quantum_computing=True,dataset_name="MNIST",sample_id=None,checkpoint_path=None,under_attack=False):
    gpu_metrics=gpu_metrics or {}
    confidence_score,logit_margin,entropy=(None,None,None)
    if logits is not None:
        confidence_score,logit_margin,entropy=get_prediction_quality(logits)
    total_energy=total_energy or 0.0; cpu_energy=cpu_energy or 0.0; gpu_energy=gpu_energy or 0.0; ram_energy=ram_energy or 0.0
    input_tokens=784; output_tokens=1; total_tokens=input_tokens+output_tokens
    joules_per_token=energy_per_token_kwh=watts_estimated=gpu_energy_pct=cpu_energy_pct=0.0
    if total_energy>0 and total_tokens>0:
        energy_per_token_kwh=round(total_energy/total_tokens,12)
        joules_per_token=round((total_energy*3_600_000)/total_tokens,6)
        if exec_time>0: watts_estimated=round((total_energy*3_600_000)/exec_time,4)
        gpu_energy_pct=round((gpu_energy/total_energy)*100,2)
        cpu_energy_pct=round((cpu_energy/total_energy)*100,2)
    correct=None
    if true_label is not None and prediction is not None:
        correct=int(prediction)==int(true_label)
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unique_device_id": DEVICE_UUID,
        "device_short_id": DEVICE_SHORT,
        "pc_name": get_hostname(),
        "collection_mode": "automated_edge",
        "sample_index": sample_index,
        "sample_id": sample_id if sample_id is not None else sample_index,
        "true_label": true_label,
        "prediction": prediction,
        "correct": correct,
        "dataset": dataset_name,
        "model_type": model_name,
        "parameters": parameters,
        "model_flops": model_flops,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "confidence_score": confidence_score,
        "logit_margin": logit_margin,
        "entropy": entropy,
        "execution_time_sec": round(exec_time,10),
        "cpu_energy_kwh": cpu_energy,
        "gpu_energy_kwh": gpu_energy,
        "ram_energy_kwh": ram_energy,
        "total_energy_kwh": total_energy,
        "total_emissions_kg": emissions_value,
        "carbon_intensity_kgco2_kwh": carbon_intensity,
        "codecarbon_version": CODECARBON_VERSION,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": round(total_tokens/exec_time,4) if exec_time>0 else None,
        "joules_per_token": joules_per_token,
        "energy_per_token_kwh": energy_per_token_kwh,
        "watts_estimated": watts_estimated,
        "gpu_energy_pct_of_total": gpu_energy_pct,
        "cpu_energy_pct_of_total": cpu_energy_pct,
        "cpu_model": CPU_MODEL_NAME,
        "cpu_architecture": CPU_ARCH,
        "cpu_core_count": CPU_CORE_COUNT,
        "cpu_thread_count": CPU_THREAD_COUNT,
        "cpu_core": CPU_CORE_COUNT,
        "cpu_thread": CPU_THREAD_COUNT,
        "cpu_tdp_w": CPU_TDP_W,
        "cpu_usage_pct": get_cpu_usage(),
        "cpu_clock_mhz": get_cpu_freq(),
        "cpu_temp_c": get_cpu_temp(),
        "cpu_power_draw_w": get_cpu_power_draw_w(),
        "cpu_cores_used": get_cpu_cores_used(),
        "gpu_model": get_gpu_name(),
        "gpu_core": GPU_CORE_COUNT,
        "gpu_thread": GPU_THREAD_COUNT,
        "gpu_driver_version": GPU_STATIC["gpu_driver_version"],
        "gpu_compute_capability": GPU_STATIC["gpu_compute_capability"],
        "gpu_power_limit_w": GPU_STATIC["gpu_power_limit_w"],
        "gpu_memory_total_mb": GPU_STATIC["gpu_memory_total_mb"],
        "gpu_power_draw_w": gpu_metrics.get("gpu_power_draw_w"),
        "gpu_utilization_pct": gpu_metrics.get("gpu_utilization_pct"),
        "gpu_temp_c": gpu_metrics.get("gpu_temp_c"),
        "gpu_memory_used_mb": gpu_metrics.get("gpu_memory_used_mb"),
        "gpu_sm_clock_mhz": gpu_metrics.get("gpu_sm_clock_mhz"),
        "gpu_memory_clock_mhz": gpu_metrics.get("gpu_memory_clock_mhz"),
        "cuda_driver_version": CUDA_DRIVER_VERSION,
        "cuda_available": torch.cuda.is_available(),
        "device_type": str(DEVICE),
        "ram_usage_pct": psutil.virtual_memory().percent,
        "memory_footprint_mb": get_memory_footprint_mb(),
        "system_ram_total_gb": SYSTEM_RAM_TOTAL_GB,
        "os_name": OS_NAME,
        "os_version": OS_VERSION,
        "os_architecture": OS_ARCHITECTURE,
        "os_full_name": OS_FULL_NAME,
        "python_version": PYTHON_VERSION,
        "torch_version": TORCH_VERSION,
        "model_accuracy": None,
        "model_precision_weighted": None,
        "model_recall_weighted": None,
        "model_f1_weighted": None,
        "quantum_computing": quantum_computing,
        "model_under_attack": int(bool(under_attack)),
        "source_qubits": N_QUBITS,
        "n_qubits": N_QUBITS,
        "fidelity": None,
        "pennylane_device": PENNYLANE_DEVICE,
        "pennylane_version": qml.__version__,
    }


def append_single_row(row, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exists=output_path.exists()
    pd.DataFrame([row]).to_csv(output_path,mode="a" if exists else "w",header=not exists,index=False)


def backfill_final_metrics(output_path, metrics):
    df=pd.read_csv(output_path)
    df["model_accuracy"]=float(metrics["accuracy"])
    df["model_precision_weighted"]=float(metrics["precision_weighted"])
    df["model_recall_weighted"]=float(metrics["recall_weighted"])
    df["model_f1_weighted"]=float(metrics["f1_weighted"])
    df.to_csv(output_path,index=False)


# ============================================================
# TEST MODEL
# ============================================================



@torch.no_grad()

def test_model(
    dataset_name,
    dataset_key,
    model,
    test_loader,
    checkpoint_path,
):
    model.eval()
    y_true=[]
    y_pred=[]
    parameters=int(sum(p.numel() for p in model.parameters()))
    model_flops=None
    model_name=f"YOLO26n-CLS-{N_QUBITS}Qubit"
    output_path=DEVICE_LOG_DIR / f"yolo26n_{dataset_key}_{N_QUBITS}qubit_test_telemetry.csv"
    if output_path.exists():
        output_path.unlink()
    sample_index=0
    for images,labels in tqdm(test_loader,desc=f"Testing {dataset_name}",unit="batch"):
        images=images.to(DEVICE)
        labels=labels.to(DEVICE)
        for local_index in range(images.size(0)):
            single_image=images[local_index:local_index+1]
            true_label=int(labels[local_index].item())
            (logits,exec_time,cpu_energy,gpu_energy,ram_energy,total_energy,emissions_value,carbon_intensity,gpu_metrics)=run_single_inference_with_telemetry(model,single_image,dataset_key)
            prediction=int(logits.argmax(dim=1).item())
            y_pred.append(prediction)
            y_true.append(true_label)
            row=build_telemetry_row(
                model_name=model_name,
                prediction=prediction,
                exec_time=exec_time,
                parameters=parameters,
                true_label=true_label,
                sample_index=sample_index,
                logits=logits,
                cpu_energy=cpu_energy,
                gpu_energy=gpu_energy,
                ram_energy=ram_energy,
                total_energy=total_energy,
                emissions_value=emissions_value,
                carbon_intensity=carbon_intensity,
                gpu_metrics=gpu_metrics,
                model_flops=model_flops,
                quantum_computing=True,
                dataset_name=dataset_name,
                sample_id=sample_index,
                checkpoint_path=checkpoint_path,
                under_attack=False,
            )
            append_single_row(row,output_path)
            sample_index+=1
    accuracy=accuracy_score(y_true,y_pred)
    precision=precision_score(y_true,y_pred,labels=list(range(N_CLASSES)),average="weighted",zero_division=0)
    recall=recall_score(y_true,y_pred,labels=list(range(N_CLASSES)),average="weighted",zero_division=0)
    f1=f1_score(y_true,y_pred,labels=list(range(N_CLASSES)),average="weighted",zero_division=0)
    metrics={
        "accuracy":float(accuracy),
        "precision_weighted":float(precision),
        "recall_weighted":float(recall),
        "f1_weighted":float(f1),
    }
    backfill_final_metrics(output_path,metrics)
    print(f"\n{dataset_name} TEST RESULTS")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Telemetry: {output_path.resolve()}")
    return metrics



# ============================================================

# MAIN

# ============================================================



def main():



    print(

        "=" * 72

    )



    print(

        "YOLO26n-CLS + 1-QUBIT "

        "PENNYLANE CLASSIFIER"

    )



    print(

        "=" * 72

    )



    print(

        f"PyTorch device   : {DEVICE}"

    )



    print(

        f"PennyLane device : "

        f"{PENNYLANE_DEVICE}"

    )



    print(

        f"Qubits           : "

        f"{N_QUBITS}"

    )



    print(

        f"Quantum layers   : "

        f"{N_Q_LAYERS}"

    )



    print(

        f"Epochs           : "

        f"{EPOCHS}"

    )



    print(

        f"Train batch      : "

        f"{TRAIN_BATCH_SIZE}"

    )



    print(

        f"Train samples    : "

        f"{TRAIN_SAMPLES if TRAIN_SAMPLES > 0 else 'ALL'}"

    )



    for (

        dataset_name,

        config,

    ) in DATASET_CONFIGS.items():



        print(

            "\n"

            + "=" * 72

        )



        print(

            f"DATASET: "

            f"{dataset_name}"

        )



        print(

            "=" * 72

        )



        (

            train_loader,

            val_loader,

            test_loader,

        ) = make_loaders(

            config[

                "dataset_class"

            ]

        )



        (

            model,

            checkpoint_path,

        ) = get_or_train_model(

            dataset_name=(

                dataset_name

            ),

            dataset_key=(

                config["key"]

            ),

            train_loader=(

                train_loader

            ),

            val_loader=(

                val_loader

            ),

        )



        print(

            f"Using checkpoint: "

            f"{checkpoint_path}"

        )
        test_model(
            dataset_name=(
                dataset_name
            ),
            dataset_key=(
                config["key"]
            ),
            model=model,
            test_loader=(
                test_loader
            ),
            checkpoint_path=(
                checkpoint_path
            ),
        )



        del model



        if torch.cuda.is_available():

            torch.cuda.empty_cache()



    print(

        "\nAll YOLO 1-qubit "

        "models complete."

    )





if __name__ == "__main__":

    main()
