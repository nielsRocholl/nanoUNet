"""Checkpoint load: strip Lightning prefix, optional EMA shadow, build net, pick ckpt file."""

from __future__ import annotations

import os

import torch
from batchgenerators.utilities.file_and_folder_operations import join

from nanounet.model.network import build_net, build_net_longi
from nanounet.plan.labels import labels_from_dataset_json

# Lightning key is Callback.state_key == __qualname__. Do not import EMACallback here.
_EMA_CB = "EMACallback"


def _strip_pl_state(sd: dict) -> dict:
    return {k[4:]: v for k, v in sd.items() if k.startswith("net.")}


def _ema_shadow(ck: dict, ckpt_path: str, net_st: dict) -> dict:
    ema_cb = (ck.get("callbacks") or {}).get(_EMA_CB)
    shadow = ema_cb.get("shadow") if isinstance(ema_cb, dict) else None
    if not shadow:
        raise SystemExit(
            f"No EMA shadow in checkpoint '{ckpt_path}'.\n"
            f"Expected Lightning callback state at callbacks/{_EMA_CB}/shadow "
            f"(written when training with --ema-decay > 0).\n"
            f"Fix: drop --ema to use raw net.* weights, or train with --ema-decay 0.999  (see docs/steps/predict.md)"
        )
    missing = [k for k in net_st if k not in shadow]
    extra = [k for k in shadow if k not in net_st]
    if missing or extra:
        raise SystemExit(
            f"EMA shadow keys do not match net.* in '{ckpt_path}'.\n"
            f"Expected the same {len(net_st)} tensors as state_dict keys stripped of 'net.'; "
            f"got {len(shadow)} ({len(missing)} missing, {len(extra)} extra).\n"
            f"Fix: this checkpoint's EMA is from a different architecture; use matching --ckpt or drop --ema  "
            f"(see docs/steps/predict.md)"
        )
    return shadow


def load_net_from_ckpt(
    ckpt_path: str, cm, dj: dict, dev: torch.device, longi: bool = False, ema: bool = False,
):
    lm = labels_from_dataset_json(dj)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    st = _strip_pl_state(sd)
    if not st:
        raise RuntimeError("no net.* keys in checkpoint")
    is_longi = longi or any(k.startswith("dwb.") for k in st)
    if ema:
        st = _ema_shadow(ck, ckpt_path, st)
    build = build_net_longi if is_longi else build_net
    net = build(cm, lm, dj, enable_deep_supervision=False)
    net.load_state_dict(st, strict=True)
    return net.to(dev).eval(), lm


def pick_checkpoint(model_dir: str, ckpt: str | None) -> str:
    name = ckpt or "last.ckpt"
    for p in (name, join(model_dir, name), join(model_dir, "checkpoints", name), join(model_dir, "finetune", name)):
        if os.path.isfile(p):
            return p
    raise SystemExit(
        f"No checkpoint '{name}' under '{model_dir}'.\n"
        f"Expected a Lightning .ckpt at checkpoints/{name} or finetune/{name}.\n"
        f"Fix: pass --ckpt <path-or-name>.ckpt  (see docs/steps/predict.md)"
    )
