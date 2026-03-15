import json
import pathlib
import sys

import numpy as np
import torch

from config import Config


class _Tee:
    """将 stdout 同时输出到终端和日志文件。"""
    def __init__(self, *files):
        self._files = files

    def write(self, data):
        for f in self._files:
            f.write(data)

    def flush(self):
        for f in self._files:
            f.flush()


def create_experiment_dir() -> pathlib.Path:
    """在 experiments/ 下以当前时间创建实验目录，格式 YYYYMMDD_HHMMSS。"""
    import datetime
    base    = pathlib.Path("experiments")
    base.mkdir(exist_ok=True)
    name    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base / name
    exp_dir.mkdir()
    return exp_dir


def save_config(exp_dir: pathlib.Path):
    """将 Config 所有超参序列化到 config.json。"""
    d = {}
    for k in sorted(vars(Config)):
        if k.startswith('_'):
            continue
        v = getattr(Config, k)
        if callable(v):
            continue
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif isinstance(v, torch.device):
            d[k] = str(v)
        else:
            d[k] = v
    with open(exp_dir / "config.json", 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
