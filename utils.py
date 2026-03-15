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
    """在 experiments/ 下创建下一个可用的 run_NNN 目录。"""
    base = pathlib.Path("experiments")
    base.mkdir(exist_ok=True)
    existing = sorted(base.glob("run_*"))
    next_id  = len(existing) + 1
    exp_dir  = base / f"run_{next_id:03d}"
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
