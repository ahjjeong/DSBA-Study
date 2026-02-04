import random
import torch
import numpy as np
import wandb
from typing import Optional
from datetime import datetime
import pytz
from omegaconf import DictConfig
import torch.nn as nn
import os
import omegaconf
from omegaconf import OmegaConf


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


def set_wandb(model: nn.Module, cfg: DictConfig) -> bool:
    if not cfg.wandb.enable:
        return False

    # key가 있을 때만 login 시도 (없으면 env var에 맡김)
    key = cfg.wandb.get("key", None)
    if key is not None:
        ok = wandb_login(key=key)
        if not ok:
            return False

    ok = init_wandb(model, cfg)
    return ok


def wandb_login(key: Optional[str] = None) -> bool:
    try:
        wandb.login(key=key)
        return True
    except Exception as e:
        print(f"[Warning] wandb login failed: {e}")
        return False


def init_wandb(model: nn.Module, cfg: DictConfig) -> bool:
    try:
        project_name = cfg.wandb.get("project", "nlp-study")
        mode = cfg.wandb.get("mode", "online")

        # run name
        if cfg.wandb.get("run_name") is None:
            now = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H-%M-%S")
            model_name = cfg.model.model_name  # ✅ 여기 수정
            lr = cfg.optimizer.lr
            run_name = f"{model_name}_lr{lr}_{now}"
        else:
            run_name = cfg.wandb.run_name

        wandb.init(
            project=project_name,
            name=run_name,
            mode=mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

        return True

    except Exception as e:
        print(f"[Warning] Failed to initialize wandb: {e}")
        return False
