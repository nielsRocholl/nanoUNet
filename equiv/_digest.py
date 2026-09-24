"""Canonical sha256 over nested python/numpy/torch values: dtype|shape|bytes per array, exact float
bits (float.hex), dict keys sorted. Two values digest equal iff they are bit-identical."""

from __future__ import annotations

import hashlib


def _feed(h, x) -> None:
    import numpy as np

    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if torch is not None and isinstance(x, torch.Tensor):
        t = x.detach().cpu().contiguous()
        h.update(f"T|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
        return
    if isinstance(x, np.ndarray):
        a = np.ascontiguousarray(x)
        h.update(f"A|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes() if a.dtype != object else repr(a.tolist()).encode())
        return
    if isinstance(x, np.generic):
        _feed(h, np.asarray(x))
        return
    if isinstance(x, dict):
        h.update(b"D{")
        for k in sorted(x, key=repr):
            h.update(repr(k).encode() + b":")
            _feed(h, x[k])
        h.update(b"}")
        return
    if isinstance(x, (list, tuple)):
        h.update(b"L[" if isinstance(x, list) else b"U[")
        for v in x:
            _feed(h, v)
        h.update(b"]")
        return
    if isinstance(x, float):
        h.update(b"F" + x.hex().encode())
        return
    if isinstance(x, (bytes, bytearray)):
        h.update(b"B" + bytes(x))
        return
    if hasattr(x, "__dataclass_fields__"):
        from dataclasses import asdict

        h.update(b"C|")  # class name excluded: K7 allows renaming config classes
        _feed(h, asdict(x))
        return
    h.update(f"R|{type(x).__name__}|{x!r}".encode())


def digest(x) -> str:
    h = hashlib.sha256()
    _feed(h, x)
    return h.hexdigest()[:24]
