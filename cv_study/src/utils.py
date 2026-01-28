import os
import random
import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_str: str) -> torch.device:
    device_str = str(device_str)
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def append_csv(path: str, header: list[str], row: list):
    ensure_dir(os.path.dirname(path))
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        f.write(",".join(map(str, row)) + "\n")