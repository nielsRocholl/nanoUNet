"""Checkpoint loaders: nnFoundationCNN or MAE encoder-only (stem zero-pad), or full supervised net."""

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
        raise TypeError(
            f"Checkpoint '{ckpt_path}' does not contain a dict-like state_dict.\n"
            f"Expected torch.load('{ckpt_path}') to return a dict, or a dict with a 'state_dict' key mapping to a dict of tensors.\n"
            f"Fix: pass a valid Lightning .ckpt to --mae-ckpt (from nanounet_pretrain)   (see nanounet/docs/steps/pretrain.md)"
        )
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


def load_full_net(seg_net: torch.nn.Module, ckpt_path: str) -> dict:
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location="cpu")
    raw = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    if not isinstance(raw, dict):
        raise TypeError(
            f"Checkpoint '{ckpt_path}' does not contain a dict-like state_dict.\n"
            f"Expected torch.load('{ckpt_path}') to return a dict, or a dict with a 'state_dict' key mapping to a dict of tensors.\n"
            f"Fix: pass a valid Lightning .ckpt to --init-weights (from a prior nanounet_train run)   (see nanounet/docs/steps/train.md)"
        )
    sd_pre = {k[4:]: v for k, v in raw.items() if k.startswith("net.") and isinstance(v, torch.Tensor)}
    sd_seg = seg_net.state_dict()
    new = {k: v for k, v in sd_pre.items() if k in sd_seg and sd_seg[k].shape == v.shape}
    miss, unex = seg_net.load_state_dict({**sd_seg, **new}, strict=False)
    _LOG.info("[init] loaded %d net tensors; missing %d unexpected %d", len(new), len(miss), len(unex))
    return {"loaded": list(new.keys()), "missing": miss, "unexpected": unex}


def load_foundation_encoder(seg_net: torch.nn.Module, ckpt_path: str, expect: int = 448) -> dict:
    """Load only `encoder.*` of the nnFoundationCNN checkpoint; decoder and seg head stay at their random init (D8).

    The stem takes [CT, prompt]; the checkpoint has 1 channel, so the prompt channel starts at zero. A kernel axis
    that is 3 in the checkpoint and 1 in the net is averaged.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd_seg = seg_net.state_dict()
    new = {}
    for k, v in ck["network_weights"].items():
        if not k.startswith("encoder.") or k not in sd_seg:
            continue
        t = sd_seg[k]
        if t.shape == v.shape:
            new[k] = v
        elif v.ndim == 5 and v.shape[2:] != t.shape[2:] and all(a == b or (b == 1 and a == 3) for a, b in zip(v.shape[2:], t.shape[2:])):
            v = v.mean(dim=[i for i in (2, 3, 4) if t.shape[i] == 1 and v.shape[i] == 3], keepdim=True)
            if t.shape == v.shape:
                new[k] = v
        elif k.startswith("encoder.stem.convs.0.") and v.ndim == 5 and v.shape[1] < t.shape[1] and v.shape[2:] == t.shape[2:]:
            w = torch.zeros_like(t)
            w[:, : v.shape[1]] = v
            new[k] = w
    if len(new) < expect:
        raise SystemExit(
            f"Only {len(new)} of {expect} encoder tensors from {ckpt_path} fit this network.\n"
            f"Expected the nnFoundationCNN ResEnc-L topology (6 stages, features 32-320, blocks 1-3-4-6-6-6).\n"
            f"Fix: use the plans written by nanounet_preprocess (nnFoundationCNN_z1p0), or pass --no-foundation"
        )
    seg_net.load_state_dict({**sd_seg, **new}, strict=False)
    _LOG.info("[foundation] loaded %d encoder tensors; decoder/seg head from scratch", len(new))
    return {"loaded": list(new.keys())}
