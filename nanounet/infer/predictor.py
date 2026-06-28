"""Checkpoint load: strip Lightning prefix, build net, pick ckpt file."""

from __future__ import annotations

import os

import torch
from batchgenerators.utilities.file_and_folder_operations import join

from nanounet.model.network import build_net, build_net_longi
from nanounet.plan.labels import labels_from_dataset_json


def _strip_pl_state(sd: dict) -> dict:
    return {k[4:]: v for k, v in sd.items() if k.startswith("net.")}


def load_net_from_ckpt(ckpt_path: str, cm, dj: dict, dev: torch.device, longi: bool = False):
    lm = labels_from_dataset_json(dj)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    st = _strip_pl_state(sd)
    if not st:
        raise RuntimeError("no net.* keys in checkpoint")
    is_longi = longi or any(k.startswith("dwb.") for k in st)
    build = build_net_longi if is_longi else build_net
    net = build(cm, lm, dj, enable_deep_supervision=False)
    net.load_state_dict(st, strict=True)
    return net.to(dev).eval(), lm


def pick_checkpoint(model_dir: str, ckpt: str | None) -> str:
    if ckpt:
        if os.path.isfile(ckpt):
            return ckpt
        p2 = join(model_dir, ckpt)
        if os.path.isfile(p2):
            return p2
        raise FileNotFoundError(ckpt)
    cdir = join(model_dir, "checkpoints")
    p = join(cdir, "last.ckpt")
    if os.path.isfile(p):
        return p
    fin = join(model_dir, "finetune", "last.ckpt")
    if os.path.isfile(fin):
        return fin
    raise FileNotFoundError(f"no checkpoint in {cdir} or finetune/")
