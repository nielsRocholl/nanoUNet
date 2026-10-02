"""nnFoundationCNN (DKFZ, CC-BY-SA-4.0) constants, download and verification.

The checkpoint is a ResEnc-L with a FIXED topology and 192^3 recommended patch, so in foundation mode the
planner does not choose the architecture: ARCH_KWARGS below is written verbatim into the plans (D9).
Weights are fetched at preprocess time and recorded in the plans' `pretrain_info`; training never touches
the network (A1). Hugging Face repo MIC-DKFZ/nnFoundationCNN, arXiv 2609.26924.
"""

from __future__ import annotations

import hashlib
import os

import torch

from nanounet.model.network import _build_class

FOUNDATION_REPO = "MIC-DKFZ/nnFoundationCNN"
FOUNDATION_FILE = "checkpoint_final.pth"
NET_CLASS = "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"
ARCH_KWARGS = {
    "n_stages": 6,
    "features_per_stage": [32, 64, 128, 256, 320, 320],
    "conv_op": "torch.nn.modules.conv.Conv3d",
    "kernel_sizes": [[3, 3, 3]] * 6,
    "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
    "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
    "conv_bias": True,
    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
    "norm_op_kwargs": {"eps": 1e-5, "affine": True},
    "dropout_op": None,
    "dropout_op_kwargs": None,
    "nonlin": "torch.nn.LeakyReLU",
    "nonlin_kwargs": {"inplace": True},
}
KW_REQUIRES_IMPORT = ["conv_op", "norm_op", "dropout_op", "nonlin"]
PATCH_SIZE = [192, 192, 192]
N_ENCODER_TENSORS = 448  # incl. the stem conv registered twice (same Parameter, two keys)


def fetch_foundation(cache_dir: str | None = None) -> str:
    """Download (or cache-hit) the checkpoint; returns its local path. cache_dir defaults to $NANOUNET_PRETRAINED."""
    cache_dir = cache_dir or os.environ.get("NANOUNET_PRETRAINED")
    if not cache_dir:
        raise SystemExit(
            "NANOUNET_PRETRAINED is not set.\n"
            "Expected a directory where pretrained weights are cached (the nnFoundationCNN checkpoint is ~410 MB).\n"
            "Fix: export NANOUNET_PRETRAINED=/nnunet_data/NanoUNet_pretrained"
        )
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(FOUNDATION_REPO, FOUNDATION_FILE, cache_dir=cache_dir)
    except Exception as e:  # nanochat-style: allow E4 (re-raised with the fix; not swallowed)
        raise SystemExit(
            f"Could not download {FOUNDATION_REPO}/{FOUNDATION_FILE} into {cache_dir}: {type(e).__name__}: {e}\n"
            f"Expected network access to huggingface.co, or the file already present in the cache.\n"
            f"Fix: huggingface-cli download {FOUNDATION_REPO} {FOUNDATION_FILE} --cache-dir {cache_dir}"
        ) from e


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_foundation(path: str) -> dict:
    """Check the checkpoint matches ARCH_KWARGS key-for-key and shape-for-shape (448/448); returns pretrain_info."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    fix = f"Fix: delete {path} and re-run nanounet_preprocess (it re-downloads {FOUNDATION_FILE})"
    for key in ("network_weights", "nnssl_adaptation_plan"):
        if key not in ck:
            raise SystemExit(  # nanochat-style: allow E1 (Fix: line is in the `fix` variable)
            f"{path} has no {key!r} key (found {sorted(ck)}).\nExpected an nnssl checkpoint.\n{fix}"
            )
    plan = ck["nnssl_adaptation_plan"]
    got = (plan["architecture_plans"]["arch_class_name"], plan["pretrain_num_input_channels"], list(plan["recommended_downstream_patchsize"]))
    if got != ("ResEncL", 1, PATCH_SIZE):
        raise SystemExit(  # nanochat-style: allow E1 (Fix: line is in the `fix` variable)
            f"{path} adaptation plan says {got}.\nExpected ('ResEncL', 1, {PATCH_SIZE}).\n{fix}"
            )
    nw, kw = _build_class(NET_CLASS, ARCH_KWARGS, KW_REQUIRES_IMPORT)
    sd = nw(input_channels=1, num_classes=2, deep_supervision=False, **kw).state_dict()
    enc = {k: v for k, v in ck["network_weights"].items() if k.startswith("encoder.")}
    bad = [k for k, v in enc.items() if k not in sd or sd[k].shape != v.shape]
    if bad or len(enc) != N_ENCODER_TENSORS:
        raise SystemExit(  # nanochat-style: allow E1 (Fix: line is in the `fix` variable)
            f"{path}: {len(enc)} encoder tensors, {len(enc) - len(bad)} matching the net by key and shape (first mismatches: {bad[:3]}).\n"
            f"Expected {N_ENCODER_TENSORS} encoder tensors, all matching.\n"
            f"This is not the nnFoundationCNN ResEnc-L topology in nanounet/model/foundation.py ARCH_KWARGS.\n{fix}"
        )
    return {
        "checkpoint_path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "repo": FOUNDATION_REPO,
        "n_encoder_tensors": len(enc),
        "citations": ck.get("citations"),
    }
