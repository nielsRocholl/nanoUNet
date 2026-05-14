"""Load MAE Lightning checkpoint into supervised ResEnc: encoder-only, stem zero-pad for prompts."""

from __future__ import annotations

import logging

import torch

_LOG = logging.getLogger("nanounet")

STEM_WEIGHT = "encoder.stem.convs.0.conv.weight"


def load_mae_encoder(seg_net: torch.nn.Module, ckpt_path: str) -> dict:
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location="cpu")
    raw = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    if not isinstance(raw, dict):
        raise TypeError(ckpt_path)
    sd_pre = {k[4:]: v for k, v in raw.items() if k.startswith("net.") and isinstance(v, torch.Tensor)}
    sd_seg = seg_net.state_dict()
    new = {}
    for k, v in sd_pre.items():
        if not k.startswith("encoder."):
            continue
        if k not in sd_seg:
            continue
        if sd_seg[k].shape == v.shape:
            new[k] = v
            continue
        if k == STEM_WEIGHT and v.shape[1] < sd_seg[k].shape[1]:
            w = torch.zeros_like(sd_seg[k])
            w[:, : v.shape[1]] = v
            new[k] = w
    merged = {**sd_seg, **new}
    miss, unex = seg_net.load_state_dict(merged, strict=False)
    _LOG.info("[MAE] loaded %d encoder tensors; missing %d unexpected %d", len(new), len(miss), len(unex))
    return {"loaded": list(new.keys()), "missing": miss, "unexpected": unex}
