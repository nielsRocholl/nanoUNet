"""NANOUNET_* path env, --config resolution, logging. Terminal UI lives in core/ui.py (shared by all projects).

`quiet_lightning_runtime`: call once before importing pytorch_lightning — warning filters,
rank_zero_info shim (litlogger noise), CUDA matmul precision high.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_LOG = logging.getLogger("nanounet")

_LIGHTNING_QUIET = False

ANISO_THRESHOLD = 3
DEFAULT_NUM_PROCESSES = 8 if "NANOUNET_DEF_N_PROC" not in os.environ else int(os.environ["NANOUNET_DEF_N_PROC"])
_REPO_ROOT = Path(__file__).resolve().parent.parent


def quiet_lightning_runtime() -> None:
    import warnings

    for msg, cat, mod in (
        (r".*pin_memory.*not supported on MPS.*", UserWarning, r"torch.utils.data.dataloader"),
        (r".*Precision 16-mixed is not supported by the model summary.*", UserWarning, None),
        (r".*LeafSpec.*", None, None),
        (r".*anonymous setting has no effect.*", UserWarning, None),
        (r".*IterableDataset.*__len__.*multi-process data loading.*", UserWarning, None),
        (r".*DataLoader will create.*worker processes in total.*", UserWarning, r"torch.utils.data.dataloader"),
        (r".*set_float32_matmul_precision.*", UserWarning, None),
    ):
        kw = {}
        if cat is not None:
            kw["category"] = cat
        if mod is not None:
            kw["module"] = mod
        warnings.filterwarnings("ignore", message=msg, **kw)
    global _LIGHTNING_QUIET
    if _LIGHTNING_QUIET:
        return
    import pytorch_lightning.utilities.rank_zero as _rz

    _orig = _rz.rank_zero_info

    def _no_litlogger_tip(*a: object, **k: object) -> None:
        if a and "litlogger" in str(a[0]).lower():
            return
        _orig(*a, **k)

    _rz.rank_zero_info = _no_litlogger_tip
    _LIGHTNING_QUIET = True
    import torch
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def resolve_user_config_path(path_str: str) -> str:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        if not p.is_file():
            raise FileNotFoundError(
                f"Config file not found: {path_str}\n"
                f"Looked as absolute, then under cwd and the nanoUNet repo root.\n"
                f"Fix: pass --config nanounet/configs/default.json (or an absolute path). See nanounet/docs/reference/config.md"
            )
        return str(p.resolve())
    for base in (Path.cwd(), _REPO_ROOT):
        cand = (base / p).resolve()
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        f"Config file not found: {path_str}\n"
        f"Looked as absolute, then under cwd and the nanoUNet repo root.\n"
        f"Fix: pass --config nanounet/configs/default.json (or an absolute path). See nanounet/docs/reference/config.md"
    )


def _env_path(name: str) -> str:
    d = os.environ.get(name)
    if not d:
        raise EnvironmentError(
            f"Required environment variable {name} is not set.\n"
            f"nanoUNet resolves dataset paths from {name}.\n"
            f"Fix: export {name}=/path/to/{name.replace('NANOUNET_','').lower()}   (see nanounet/docs/index.md)"
        )
    return d


def raw_dir() -> str:
    return _env_path("NANOUNET_RAW")


def preprocessed_dir() -> str:
    return _env_path("NANOUNET_PREPROCESSED")


def results_dir() -> str:
    return _env_path("NANOUNET_RESULTS")


def setup_logging() -> None:
    if not _LOG.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        _LOG.addHandler(h)
        _LOG.setLevel(logging.INFO)
