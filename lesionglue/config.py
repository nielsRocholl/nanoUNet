"""Canonical r9_base training config: dataclass + JSON load/save."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

from tracking.common import load_json

CKPT_MONITOR = "val_match_score_ema"


@dataclass
class Config:
    max_steps: int = 8000
    val_check_steps: int = 250
    warmup_steps: int = 1000
    early_stop_patience: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 8
    val_batch_size: int = 1
    num_workers: int = 2
    seed: int = 0
    d: int = 128
    layers: int = 4
    heads: int = 4
    geo: bool = True
    geo_knn: int = 3
    dropout: float = 0.2
    sinkhorn_w: float = 1.0
    pair_w: float = 0.1
    nce_w: float = 0.3
    dust_w: float = 0.30
    dust_pos_w: float = 1.0
    nce_tau: float = 0.1
    sinkhorn_iters: int = 20
    fu_jitter: float = 0.3
    p_drop_fu: float = 0.10
    p_drop_bl: float = 0.10
    k_intra: int = 8
    ema_decay: float = 0.999
    ema_start_step: int = 1000
    dust_tau: float = 0.2
    val_score_ema_beta: float = 0.3
    n_folds: int = 5
    cv_seed: int = 0


def load_config(path: str | Path) -> Config:
    raw = load_json(path)
    known = {f.name for f in fields(Config)}
    extra = set(raw) - known
    if extra:
        raise ValueError(f"unknown config keys: {sorted(extra)}")
    return Config(**{k: raw[k] for k in known if k in raw})


def dump_config(cfg: Config, path: str | Path) -> None:
    from tracking.common import dump_json

    dump_json(path, asdict(cfg))
