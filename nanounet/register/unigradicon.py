"""Deep-learning BL->FU backend: pretrained uniGradICON foundation model.

preprocess() clips/normalizes CT to [0,1] for the network only; the returned transform is applied to
the ORIGINAL (unpreprocessed) BL image/seg/id-map via elastix.resample_to so HU and labels survive.
Native instance optimization (`io_iterations`) is this backend's refinement — it replaces the
per-lesion elastix `refine` step, which only runs for the elastix backend. GPU preferred; CPU is slow
(IO is `io_iterations` backward passes through a 3D UNet at 175^3).
"""

from __future__ import annotations

import os
import urllib.error

from nanounet.register.elastix import resample_to

WEIGHTS_URL = "https://github.com/uncbiag/uniGradICON/releases/download/unigradicon_weights/Step_2_final.trch"
WEIGHTS_PATH = os.environ.get(
    "NANOUNET_UNIGRADICON_WEIGHTS",
    os.path.expanduser("~/.cache/nanounet/unigradicon/Step_2_final.trch"),
)

_MODEL = None


def get_model():
    """Load + cache the pretrained network. Call once at startup so a missing package or a blocked
    weight download fails at t=0 with a fix, not with a raw traceback mid-batch (nanochat R15/E1)."""
    global _MODEL
    if _MODEL is None:
        try:
            from unigradicon import get_unigradicon
        except ImportError as e:
            raise ImportError(
                "uniGradICON backend requested but `unigradicon` is not installed.\n"
                "Fix: uv sync   (adds unigradicon>=1.0.4 from pyproject)"
            ) from e
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        try:
            _MODEL = get_unigradicon(weights_location=WEIGHTS_PATH)
        except urllib.error.URLError as e:
            raise FileNotFoundError(
                f"uniGradICON weights missing at {WEIGHTS_PATH} and the download failed ({e}).\n"
                f"Pre-stage the file there, or set NANOUNET_UNIGRADICON_WEIGHTS to an existing copy.\n"
                f"Fix: curl -L -o {WEIGHTS_PATH} {WEIGHTS_URL}"
            ) from e
    return _MODEL


def warp_pair(
    fu: "itk.Image",
    bl: "itk.Image",
    bl_seg: "itk.Image",
    bl_ids: "itk.Image",
    *,
    io_iterations: int,
    verbose: bool = False,
) -> tuple["itk.Image", "itk.Image", "itk.Image"]:
    import icon_registration.itk_wrapper as itk_wrapper
    import itk
    from unigradicon import preprocess

    fu_pp = preprocess(fu, "ct")
    bl_pp = preprocess(bl, "ct")
    steps = io_iterations if io_iterations and io_iterations > 0 else None
    phi_AB, _ = itk_wrapper.register_pair(get_model(), bl_pp, fu_pp, finetune_steps=steps)

    warped_img = resample_to(bl, phi_AB, fu, default=-1000.0)
    nn_seg = itk.NearestNeighborInterpolateImageFunction.New(bl_seg)
    warped_seg = resample_to(bl_seg, phi_AB, fu, default=0.0, interp=nn_seg)
    nn_ids = itk.NearestNeighborInterpolateImageFunction.New(bl_ids)
    warped_ids = resample_to(bl_ids, phi_AB, fu, default=0.0, interp=nn_ids)
    return warped_img, warped_seg, warped_ids
