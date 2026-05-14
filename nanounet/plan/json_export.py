"""Coerce numpy/torch scalars for ``json.dump`` (nnU-Net ``recursive_fix_for_json_export`` subset)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


def _fix_iterable(seq: Iterable, output_type):
    out: list = []
    for i in seq:
        if isinstance(i, (np.floating, float)):
            out.append(float(i))
        elif isinstance(i, (np.integer, int)):
            out.append(int(i))
        elif isinstance(i, np.bool_):
            out.append(bool(i))
        else:
            out.append(i)
    return output_type(out)


def recursive_fix_for_json_export(d: dict) -> None:
    keys = list(d.keys())
    for k in keys:
        if isinstance(k, (np.int64, np.int32, np.int8, np.uint8)):
            tmp = d[k]
            del d[k]
            d[int(k)] = tmp
            k = int(k)
        v = d[k]
        if isinstance(v, dict):
            recursive_fix_for_json_export(v)
        elif isinstance(v, np.ndarray):
            assert v.ndim == 1
            d[k] = _fix_iterable(v, list)
        elif isinstance(v, np.bool_):
            d[k] = bool(v)
        elif isinstance(v, (np.int64, np.int32, np.int8, np.uint8)):
            d[k] = int(v)
        elif isinstance(v, (np.float32, np.float64, np.float16)):
            d[k] = float(v)
        elif isinstance(v, list):
            d[k] = _fix_iterable(v, list)
        elif isinstance(v, tuple):
            d[k] = _fix_iterable(v, tuple)
        elif isinstance(v, torch.device):
            d[k] = str(v)
