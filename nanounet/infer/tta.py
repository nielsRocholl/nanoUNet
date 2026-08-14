"""Test-time mirroring: identity + all axis combinations, fused into as few net() calls as possible."""

from __future__ import annotations

import itertools

import torch

# Cap concatenated patches per net() so TTA on a large seed batch fits a 10 GB GPU.
# B=1 still fuses all 8 mirrors; B=8 falls back to 8 calls of size B (old TTA occupancy).
_MAX_CAT = 8


@torch.inference_mode()
def predict_batch_with_tta(net: torch.nn.Module, x: torch.Tensor, use_mirroring: bool, mirror_axes=(0, 1, 2)):
    """x: (B, C, Z, Y, X) -> (B, n_heads, Z, Y, X), mirror-averaged."""
    if not use_mirroring:
        return net(x)
    ma = [m + 2 for m in mirror_axes]
    axes_list: list[tuple[int, ...]] = [()]
    axes_list.extend(c for i in range(len(ma)) for c in itertools.combinations(ma, i + 1))
    n, b = len(axes_list), x.shape[0]
    chunk = max(1, min(n, _MAX_CAT // max(b, 1)))
    acc = None
    for i in range(0, n, chunk):
        views = [x if not ax else torch.flip(x, ax) for ax in axes_list[i : i + chunk]]
        parts = net(torch.cat(views, 0)).split(b, 0)
        for ax, p in zip(axes_list[i : i + chunk], parts):
            u = p if not ax else torch.flip(p, ax)
            acc = u if acc is None else acc + u
    return acc / n
