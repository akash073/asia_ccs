"""Hybrid YOLO26n classification + PennyLane 4-layer VQC.
Supports MNIST, FashionMNIST, KMNIST, CIFAR100 and TinyImageNet200.
Auto-trains missing dataset-specific checkpoints; evaluates official held-out splits.
Keeps original unified telemetry schema and resumes incomplete CSVs.

Quantum circuit is simulated on a classical CPU (not a physical QPU).
"""
from __future__ import annotations
import os, sys, time, random, hashlib, platform, socket
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ultralytics import YOLO

SEED=int(os.getenv('RANDOM_SEED','42'))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DATA_ROOT=Path(os.getenv('DATA_ROOT','./data'))
TINY_ROOT=Path(os.getenv('TINY_IMAGENET_ROOT',str(DATA_ROOT/'tiny-imagenet-200')))
CHECKPOINT_DIR=Path(os.getenv('CHECKPOINT_DIR','./quantum_checkpoints'))
CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)
EPOCHS=int(os.getenv('TRAIN_EPOCHS','10'))
TRAIN_SAMPLES=int(os.getenv('TRAIN_SAMPLES','2000'))  # 0 means all
NUM_TEST_SAMPLES=int(os.getenv('NUM_TEST_SAMPLES','10'))  # 0 means all
TRAIN_BATCH_SIZE=int(os.getenv('TRAIN_BATCH_SIZE','8'))
NUM_WORKERS=int(os.getenv('NUM_WORKERS','0'))
LR=float(os.getenv('LEARNING_RATE','0.001'))
WEIGHT_DECAY=float(os.getenv('WEIGHT_DECAY','0.0005'))
IMG_SIZE=int(os.getenv('IMG_SIZE','64'))
N_QUBITS=int(os.getenv('N_QUBITS','4'))
N_Q_LAYERS=4
ACTIVE_CLASSES=10
PENNYLANE_DEVICE=os.getenv('PENNYLANE_DEVICE','default.qubit')
FORCE_RETRAIN=os.getenv('FORCE_RETRAIN','0')=='1'
DEVICE=torch.device('cpu')  # default.qubit + torch interface are CPU based; stable telemetry
SELECTED_DATASETS=[x.strip() for x in os.getenv('DATASETS','MNIST,FashionMNIST,KMNIST,CIFAR100,TinyImageNet200').split(',') if x.strip()]
DATASET_CONFIGS={
 'MNIST':(datasets.MNIST,'mnist',10),
 'FashionMNIST':(datasets.FashionMNIST,'fashionmnist',10),
 'KMNIST':(datasets.KMNIST,'kmnist',10),
 'CIFAR100':(datasets.CIFAR100,'cifar100',100),
 'TinyImageNet200':(None,'tinyimagenet200',200),
}
assert N_QUBITS>=1 and TRAIN_BATCH_SIZE>=1 and IMG_SIZE>=32
assert EPOCHS>=1 and NUM_TEST_SAMPLES>=0 and TRAIN_SAMPLES>=0
if not set(SELECTED_DATASETS).issubset(DATASET_CONFIGS):
 raise ValueError(f'Invalid datasets: {SELECTED_DATASETS}')

class TinyImageNetDataset(Dataset):
 def __init__(self, split, transform):
  wnids_file=TINY_ROOT/'wnids.txt'
  if not wnids_file.is_file(): raise FileNotFoundError(f'Missing {wnids_file}')
  wnids=sorted(x.strip() for x in wnids_file.read_text().splitlines() if x.strip())
  if len(wnids)!=200: raise ValueError('Expected exactly 200 Tiny ImageNet WNIDs')
  lut={w:i for i,w in enumerate(wnids)}
  self.samples=[]; self.transform=transform
  if split=='train':
   for w in wnids:
    folder=TINY_ROOT/'train'/w/'images'
    for p in sorted(folder.glob('*')):
     if p.suffix.lower() in ('.jpg','.jpeg','.png'):
      self.samples.append((p,lut[w]))
  elif split=='test':
   ann=TINY_ROOT/'val'/'val_annotations.txt'
   if not ann.is_file(): raise FileNotFoundError(f'Missing {ann}')
   for line in ann.read_text().splitlines():
    fields=line.split()
    if len(fields)>=2 and fields[1] in lut:
     p=TINY_ROOT/'val'/'images'/fields[0]
     if p.is_file(): self.samples.append((p,lut[fields[1]]))
  else: raise ValueError(split)
  if not self.samples: raise FileNotFoundError(f'No Tiny ImageNet images for {split}')
 def __len__(self): return len(self.samples)
 def __getitem__(self, idx):
  p,y=self.samples[idx]
  with Image.open(p) as im: x=self.transform(im.convert('RGB'))
  return x,y

def get_dataset(name, train):
 cfg=DATASET_CONFIGS[name]
 transform=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.Lambda(lambda im: im.convert('RGB')),transforms.ToTensor()])
 if name=='TinyImageNet200': return TinyImageNetDataset('train' if train else 'test',transform)
 return cfg[0](root=str(DATA_ROOT),train=train,download=True,transform=transform)

def make_loaders(name):
 all_train=get_dataset(name,True)
 if TRAIN_SAMPLES and TRAIN_SAMPLES<len(all_train):
  rng=np.random.default_rng(SEED)
  idx=rng.choice(len(all_train),size=TRAIN_SAMPLES,replace=False).tolist()
  all_train=Subset(all_train,idx)
 n=len(all_train)
 if n<3: raise ValueError('Need at least 3 training images')
 n_val=max(1,round(n*0.1))
 train,val=torch.utils.data.random_split(all_train,[n-n_val,n_val],generator=torch.Generator().manual_seed(SEED))
 test=get_dataset(name,False)
 if NUM_TEST_SAMPLES: test=Subset(test,range(min(NUM_TEST_SAMPLES,len(test))))
 kw=dict(batch_size=TRAIN_BATCH_SIZE,num_workers=NUM_WORKERS)
 return (DataLoader(train,shuffle=True,**kw),DataLoader(val,shuffle=False,**kw),
         DataLoader(test,shuffle=False,**kw))

# Each qnode is executed sample-by-sample, with four trainable layers.
def make_qnode(n_qubits):
 dev=qml.device(PENNYLANE_DEVICE,wires=n_qubits)
 @qml.qnode(dev,interface='torch',diff_method='best')
 def circuit(x,weights):
  for wire in range(n_qubits):
   qml.RY(x[wire],wires=wire)
   qml.RZ(x[wire],wires=wire)
  for layer in range(N_Q_LAYERS):
   for wire in range(n_qubits):
    qml.Rot(*weights[layer,wire],wires=wire)
   if n_qubits>1:
    # Neighbor chain, no duplicate reverse CNOT for the two-qubit case.
    for wire in range(n_qubits-1):
     qml.CNOT(wires=[wire,wire+1])
  return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]
 return circuit

class QuantumYOLOHead(nn.Module):
 def __init__(self,in_features,num_classes,n_qubits=N_QUBITS):
  super().__init__()
  self.n_qubits=n_qubits
  self.feature_to_qubits=nn.Linear(in_features,n_qubits)
  self.q_weights=nn.Parameter(torch.randn(N_Q_LAYERS,n_qubits,3)*0.01)
  self.classifier=nn.Linear(n_qubits,num_classes)
  self.qnode=make_qnode(n_qubits)
 def forward(self,x):
  if x.ndim==1: x=x.unsqueeze(0)
  angles=torch.tanh(self.feature_to_qubits(x))*torch.pi
  rows=[]
  for one in angles:
   out=self.qnode(one,self.q_weights)
   if isinstance(out,torch.Tensor): features=out.reshape(-1)
   else: features=torch.stack(list(out)).reshape(-1)
   rows.append(features.to(dtype=self.classifier.weight.dtype))
  return self.classifier(torch.stack(rows,0))

def get_last_linear(model):
 matches=[(name,module) for name,module in model.named_modules() if isinstance(module,nn.Linear)]
 if not matches: raise RuntimeError('YOLO classification model has no nn.Linear; check Ultralytics version')
 name,layer=matches[-1]
 parent=model
 for key in name.split('.')[:-1]: parent=parent[int(key)] if key.isdecimal() else getattr(parent,key)
 return parent,name.split('.')[-1],layer

def build_model(n_classes):
 yolo=YOLO('yolo26n-cls.yaml')
 model=yolo.model
 parent,key,old=get_last_linear(model)
 new=QuantumYOLOHead(old.in_features,n_classes)
 if key.isdecimal(): parent[int(key)]=new
 else: setattr(parent,key,new)
 return model.to(DEVICE)

def get_logits(output,n_classes):
 if isinstance(output,torch.Tensor):
  if output.ndim==2 and output.shape[1]==n_classes: return output
 if isinstance(output,(tuple,list)):
  # Ultralytics Classify.forward(eval) generally returns (probs, logits).
  tensors=[x for x in output if isinstance(x,torch.Tensor) and x.ndim==2 and x.shape[1]==n_classes]
  if tensors: return tensors[-1]  # logits preferred, not probabilities
 if isinstance(output,dict):
  for key in ('logits','preds','output'):
   if key in output: return get_logits(output[key],n_classes)
 raise RuntimeError(f'Unexpected YOLO logits format: {type(output)}')

def checkpoint_path(dataset_key):
 return CHECKPOINT_DIR/f'hybrid_yolo26n_{dataset_key}_{N_QUBITS}q_{N_Q_LAYERS}layers_{IMG_SIZE}px_v2.pth'

def load_or_train(name,key,n_classes,train_loader,val_loader):
 path=checkpoint_path(key)
 model=build_model(n_classes)
 if path.exists() and not FORCE_RETRAIN:
  try: ck=torch.load(path,map_location='cpu',weights_only=True)
  except TypeError: ck=torch.load(path,map_location='cpu')
  if ck.get('n_classes')!=n_classes or ck.get('n_qubits')!=N_QUBITS:
   raise RuntimeError(f'Checkpoint shape mismatch: {path}')
  model.load_state_dict(ck['state_dict'],strict=True)
  print(f'Loaded {path}')
  return model.eval(),path
 loss_fn=nn.CrossEntropyLoss()
 opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
 best=-1.0
 for epoch in range(1,EPOCHS+1):
  model.train()
  train_ok=train_n=0
  for x,y in tqdm(train_loader,desc=f'{name} training {epoch}/{EPOCHS}'):
   x=x.to(DEVICE); y=y.to(DEVICE)
   opt.zero_grad(set_to_none=True)
   logits=get_logits(model(x),n_classes)
   loss=loss_fn(logits,y)
   loss.backward(); opt.step()
   train_ok+=int((logits.argmax(1)==y).sum()); train_n+=len(y)
  model.eval(); val_ok=val_n=0
  with torch.no_grad():
   for x,y in val_loader:
    logits=get_logits(model(x.to(DEVICE)),n_classes)
    val_ok+=int((logits.argmax(1)==y.to(DEVICE)).sum()); val_n+=len(y)
  val_acc=val_ok/max(val_n,1)
  print(f'{name} epoch={epoch} train_acc={train_ok/max(train_n,1):.4f} val_acc={val_acc:.4f}')
  if val_acc>best:
   best=val_acc
   torch.save({'state_dict':model.state_dict(),'n_classes':n_classes,'n_qubits':N_QUBITS,'epoch':epoch,'val_acc':val_acc},path)
 try: ck=torch.load(path,map_location='cpu',weights_only=True)
 except TypeError: ck=torch.load(path,map_location='cpu')
 model.load_state_dict(ck['state_dict'],strict=True)
 return model.eval(),path

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
            project_name=f"yolo_{N_QUBITS}qubit_{dataset_key}_test",
            output_dir=str(energy_dir),
            output_file=f"codecarbon_yolo_{N_QUBITS}qubit_{dataset_key}.csv",
            log_level="error",
            save_to_file=True,
        )
        tracker.start()
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t0=time.perf_counter()
    output=model(image)
    logits=get_logits(output,ACTIVE_CLASSES)
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
    input_tokens=IMG_SIZE*IMG_SIZE*3; output_tokens=1; total_tokens=input_tokens+output_tokens
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
# TEST / RESUME, preserves all telemetry columns from uploaded code
# ============================================================
def backfill_from_csv(output_path,n_classes):
 df=pd.read_csv(output_path)
 if df.empty: return
 good=df.dropna(subset=['true_label','prediction'])
 if good.empty: return
 yt=good.true_label.astype(int).to_numpy(); yp=good.prediction.astype(int).to_numpy()
 metrics={'accuracy':accuracy_score(yt,yp),
          'precision_weighted':precision_score(yt,yp,labels=list(range(n_classes)),average='weighted',zero_division=0),
          'recall_weighted':recall_score(yt,yp,labels=list(range(n_classes)),average='weighted',zero_division=0),
          'f1_weighted':f1_score(yt,yp,labels=list(range(n_classes)),average='weighted',zero_division=0)}
 for key,value in [('model_accuracy','accuracy'),('model_precision_weighted','precision_weighted'),('model_recall_weighted','recall_weighted'),('model_f1_weighted','f1_weighted')]: df[key]=float(metrics[value])
 df.to_csv(output_path,index=False)
 print('Metrics:',metrics)

def test_model(name,key,n_classes,model,test_loader,path):
 model.eval()
 label=f'HybridYOLO26n-{N_QUBITS}Q'
 out=DEVICE_LOG_DIR/f'yolo26n_{key}_{N_QUBITS}qubit_{IMG_SIZE}px_v2_test_telemetry.csv'
 # Skip rows by stable test-set sample index, not row counts.
 done=set()
 if out.is_file():
  df=pd.read_csv(out)
  if not {'sample_index','model_type'}.issubset(df.columns):
   raise RuntimeError(f'Existing telemetry schema mismatch: {out}')
  done=set(df.loc[df.model_type==label,'sample_index'].dropna().astype(int))
 param_count=sum(p.numel() for p in model.parameters())
 i=0
 for x,y in tqdm(test_loader,desc=f'{name} telemetry'):
  for k in range(len(y)):
   idx=i;i+=1
   if idx in done: continue
   one=x[k:k+1].to(DEVICE)
   label_true=int(y[k])
   with torch.no_grad():
    (logits,elapsed,cpu_e,gpu_e,ram_e,total_e,emission,ci,gpu)=run_single_inference_with_telemetry(model,one,key)
   predicted=int(logits.argmax(dim=1).item())
   row=build_telemetry_row(model_name=label,prediction=predicted,exec_time=elapsed,
    parameters=param_count,true_label=label_true,sample_index=idx,logits=logits,
    cpu_energy=cpu_e,gpu_energy=gpu_e,ram_energy=ram_e,total_energy=total_e,
    emissions_value=emission,carbon_intensity=ci,gpu_metrics=gpu,model_flops=None,
    quantum_computing=True,dataset_name=name,sample_id=idx,checkpoint_path=path)
   append_single_row(row,out)
   done.add(idx)
 if out.is_file():
  backfill_from_csv(out,n_classes)
  print('Saved:',out)

def main():
 print('Hybrid YOLO26n + 4-layer simulated VQC',N_QUBITS,'qubits; device:',DEVICE)
 for name in SELECTED_DATASETS:
  global ACTIVE_CLASSES
  _,key,n_classes=DATASET_CONFIGS[name]
  ACTIVE_CLASSES=n_classes
  print('\n======',name,'classes',n_classes,'======')
  train_loader,val_loader,test_loader=make_loaders(name)
  model,path=load_or_train(name,key,n_classes,train_loader,val_loader)
  test_model(name,key,n_classes,model,test_loader,path)
  del model
 print('Done')

if __name__=='__main__': main()
