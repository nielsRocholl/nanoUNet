"""Test-time mirroring: identity + all axis combinations, fused into as few net() calls as possible.

Cat size from free VRAM after logits_acc. Bytes/fwd = incremental B=2−B=1 after a warmup forward. Silent; CLI reads last_*.
"""

from __future__ import annotations

import itertools

import torch

HEADROOM = 0.5
MAX_CAT = 256

_BYTES_PER: int | None = None
last_max_cat: int | None = None
last_bytes_per: int | None = None
last_free: int | None = None


def max_cat(net: torch.nn.Module, x: torch.Tensor, dev: torch.device) -> int:
    """Patches that fit in one net(). Probes bytes/fwd once; then mem_get_info."""
    global _BYTES_PER, last_max_cat, last_bytes_per, last_free
    if dev.type != "cuda":
        last_max_cat = MAX_CAT
        last_bytes_per = last_free = None
        return MAX_CAT
    if _BYTES_PER is None:
        x1 = x[:1]
        y = net(x1)
        del y
        torch.cuda.synchronize(dev)

        def _peak(b):
            xb = x1 if b == 1 else x1.expand(b, *x1.shape[1:]).contiguous()
            torch.cuda.reset_peak_memory_stats(dev)
            base = torch.cuda.memory_allocated(dev)
            y = net(xb)
            torch.cuda.synchronize(dev)
            peak = torch.cuda.max_memory_allocated(dev) - base
            del y
            if b > 1:
                del xb
            return peak

        _BYTES_PER = max(_peak(2) - _peak(1), 1)
        assert _BYTES_PER > 0
        torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(dev)
    last_max_cat = max(1, min(MAX_CAT, int(free * HEADROOM / _BYTES_PER)))
    last_bytes_per, last_free = _BYTES_PER, free
    return last_max_cat


def cat_status() -> str | None:
    if last_max_cat is None:
        return None
    if last_bytes_per is not None and last_free is not None:
        return f"TTA max_cat={last_max_cat} (probed {last_bytes_per / 1e6:.0f}MB/fwd, {last_free / 1e9:.1f}GB free)"
    return f"TTA max_cat={last_max_cat}"


@torch.inference_mode()
def predict_batch_with_tta(net: torch.nn.Module, x: torch.Tensor, use_mirroring: bool, mirror_axes=(0, 1, 2)):
    """x: (B, C, Z, Y, X) -> (B, n_heads, Z, Y, X), mirror-averaged."""
    if not use_mirroring:
        return net(x)
    ma = [m + 2 for m in mirror_axes]
    axes_list: list[tuple[int, ...]] = [()]
    axes_list.extend(c for i in range(len(ma)) for c in itertools.combinations(ma, i + 1))
    n, b = len(axes_list), x.shape[0]
    chunk = max(1, min(n, max_cat(net, x, x.device) // max(b, 1)))
    acc = None
    for i in range(0, n, chunk):
        views = [x if not ax else torch.flip(x, ax) for ax in axes_list[i : i + chunk]]
        parts = net(torch.cat(views, 0)).split(b, 0)
        for ax, p in zip(axes_list[i : i + chunk], parts):
            u = p if not ax else torch.flip(p, ax)
            acc = u if acc is None else acc + u
    return acc / n
