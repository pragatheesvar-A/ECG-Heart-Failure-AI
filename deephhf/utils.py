from __future__ import annotations

import json
import logging
import random
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(output_dir: str | Path, name: str = "deephhf") -> logging.Logger:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(directory / "run.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def save_json(value: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, default=float)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
