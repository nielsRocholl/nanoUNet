"""ITK/elastix thread selection for local (M-series) and cluster runs."""

from __future__ import annotations

import os

import itk


def resolve_threads(spec: str | int | None = "auto") -> int:
    if spec is None or spec == "auto":
        env = os.environ.get("NANOUNET_REG_THREADS")
        if env:
            return max(1, int(env))
        return itk.MultiThreaderBase.New().GetGlobalDefaultNumberOfThreads()
    if spec == "all":
        return itk.MultiThreaderBase.New().GetGlobalMaximumNumberOfThreads()
    n = int(spec)
    if n < 1:
        raise ValueError(f"threads must be >= 1, got {n}")
    return n


def apply_threads(n: int) -> int:
    itk.set_nthreads(n)
    return n
