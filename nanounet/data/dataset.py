"""List case IDs from preprocessed folder (Blosc2)."""

from __future__ import annotations

import os


def case_ids(folder: str) -> list[str]:
    suf = ".b2nd"
    return sorted(
        f[: -len(suf)] for f in os.listdir(folder) if f.endswith(suf) and not f.endswith("_seg.b2nd")
)
