"""Shared deterministic execution settings."""

import os
import random

import numpy as np
import torch

from .config import Config
from .setup_check import select_device


def initialize(config: Config) -> torch.device:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return device
