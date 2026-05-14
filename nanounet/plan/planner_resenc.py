"""ResidualEncoder 3D ``3d_fullres`` block: VRAM loop, topology, batch size (nnU-Net ResEnc presets)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

from nanounet.data.resampling import resample_data_or_seg_to_shape
from nanounet.model.network import estimate_conv_feature_map_size
from nanounet.plan.planner_topology import get_pool_and_conv_props


@dataclass(frozen=True)
class ResEncPlannerPreset:
    plans_identifier: str
    default_vram_gb: float
    reference_val_corresp_gb: float
    reference_val_3d: float
    max_dataset_covered: float = 1.0
    max_feature_channels: int = 320
    enc_blocks: Tuple[int, ...] | None = None
    dec_blocks: Tuple[int, ...] | None = None


ENC_BLOCKS = (1, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)
DEC_BLOCKS = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
TINY_ENC_BLOCKS = (1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
TINY_DEC_BLOCKS = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

PRESETS = {
    "nnUNetPlannerResEncTiny": ResEncPlannerPreset(
        "nnUNetResEncUNetTinyPlans",
        8.0,
        8.0,
        80_000_000,
        1.0,
        max_feature_channels=64,
        enc_blocks=TINY_ENC_BLOCKS,
        dec_blocks=TINY_DEC_BLOCKS,
    ),
    "nnUNetPlannerResEncM": ResEncPlannerPreset("nnUNetResEncUNetMPlans", 8.0, 8.0, 680_000_000, 1.0),
    "nnUNetPlannerResEncL": ResEncPlannerPreset("nnUNetResEncUNetLPlans", 24.0, 24.0, 2_100_000_000, 1.0),
    "nnUNetPlannerResEncXL": ResEncPlannerPreset("nnUNetResEncUNetXLPlans", 40.0, 40.0, 3_600_000_000, 1.0),
}

REF_BS_3D = 2
MIN_BATCH = 2
MIN_EDGE = 4
NET_CLASS = ResidualEncoderUNet.__module__ + "." + ResidualEncoderUNet.__name__


def resenc_3d_fullres_plan(
    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
    median_shape: Union[np.ndarray, Tuple[int, ...], List[int]],
    data_identifier: str,
    approximate_n_voxels_dataset: float,
    dataset_json: dict,
    normalization_schemes: list[str],
    use_mask_for_norm: list[bool],
    preprocessor_name: str,
    vram_target_gb: float,
    preset: ResEncPlannerPreset,
    max_patch_size_in_voxels: Tuple[int, int, int] | None,
    cache: dict,
    patch_edge: int = 256,
) -> dict:
    feat_cap = preset.max_feature_channels
    enc_src = ENC_BLOCKS if preset.enc_blocks is None else preset.enc_blocks
    dec_src = DEC_BLOCKS if preset.dec_blocks is None else preset.dec_blocks

    def _feat_stages(n_stg: int) -> Tuple[int, ...]:
        return tuple(min(feat_cap, 32 * 2**i) for i in range(n_stg))

    def _key(ps, strides):
        return str(ps) + "_" + str(strides)

    spacing = list(spacing)
    median_shape = list(median_shape)
    assert all(x > 0 for x in spacing) and len(spacing) == 3
    modalities = dataset_json["channel_names"] if "channel_names" in dataset_json else dataset_json["modality"]
    n_in = len(modalities.keys())
    n_out = len(dataset_json["labels"].keys())
    cop = convert_dim_to_conv_op(3)
    norm = get_matching_instancenorm(cop)
    tmp = 1 / np.array(spacing)
    pe = float(patch_edge)
    initial_patch_size = [round(i) for i in tmp * (pe**3 / np.prod(tmp)) ** (1 / 3)]
    initial_patch_size = [min(i, j) for i, j in zip(initial_patch_size, median_shape[: len(spacing)])]
    if max_patch_size_in_voxels is not None:
        m = list(max_patch_size_in_voxels)[: len(initial_patch_size)]
        initial_patch_size = [min(i, x) for i, x in zip(initial_patch_size, m)]
    _, pool_ks, conv_ks, patch_size, must_div = get_pool_and_conv_props(
        spacing, initial_patch_size, MIN_EDGE, 999999
    )
    n_stages = len(pool_ks)
    arch = {
        "network_class_name": NET_CLASS,
        "arch_kwargs": {
            "n_stages": n_stages,
            "features_per_stage": list(_feat_stages(n_stages)),
            "conv_op": cop.__module__ + "." + cop.__name__,
            "kernel_sizes": list(conv_ks),
            "strides": list(pool_ks),
            "n_blocks_per_stage": list(enc_src[:n_stages]),
            "n_conv_per_stage_decoder": list(dec_src[: n_stages - 1]),
            "conv_bias": True,
            "norm_op": norm.__module__ + "." + norm.__name__,
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
    }
    key = _key(patch_size, pool_ks)
    if key in cache:
        est = cache[key]
    else:
        est = estimate_conv_feature_map_size(
            patch_size, n_in, n_out, arch["network_class_name"], arch["arch_kwargs"], arch["_kw_requires_import"]
        )
        cache[key] = est
    ref = preset.reference_val_3d * (vram_target_gb / preset.reference_val_corresp_gb)
    while est > ref:
        axis = np.argsort([i / j for i, j in zip(patch_size, median_shape[: len(spacing)])])[-1]
        ps = list(patch_size)
        t = deepcopy(ps)
        t[axis] -= must_div[axis]
        _, _, _, _, must_div = get_pool_and_conv_props(spacing, t, MIN_EDGE, 999999)
        ps[axis] -= must_div[axis]
        _, pool_ks, conv_ks, patch_size, must_div = get_pool_and_conv_props(spacing, ps, MIN_EDGE, 999999)
        n_stages = len(pool_ks)
        arch["arch_kwargs"].update(
            {
                "n_stages": n_stages,
                "kernel_sizes": list(conv_ks),
                "strides": list(pool_ks),
                "features_per_stage": list(_feat_stages(n_stages)),
                "n_blocks_per_stage": list(enc_src[:n_stages]),
                "n_conv_per_stage_decoder": list(dec_src[: n_stages - 1]),
            }
        )
        key = _key(patch_size, pool_ks)
        if key in cache:
            est = cache[key]
        else:
            est = estimate_conv_feature_map_size(
                patch_size, n_in, n_out, arch["network_class_name"], arch["arch_kwargs"], arch["_kw_requires_import"]
            )
            cache[key] = est
    batch_size = round((ref / est) * REF_BS_3D)
    bs5 = round(approximate_n_voxels_dataset * preset.max_dataset_covered / np.prod(patch_size, dtype=np.float64))
    batch_size = max(min(batch_size, bs5), MIN_BATCH)
    rdk = {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None}
    rsk = {"is_seg": True, "order": 1, "order_z": 0, "force_separate_z": None}
    rpk = {"is_seg": False, "order": 1, "order_z": 0, "force_separate_z": None}
    return {
        "data_identifier": data_identifier,
        "preprocessor_name": preprocessor_name,
        "batch_size": batch_size,
        "patch_size": list(patch_size),
        "median_image_size_in_voxels": median_shape,
        "spacing": spacing,
        "normalization_schemes": normalization_schemes,
        "use_mask_for_norm": use_mask_for_norm,
        "resampling_fn_data": resample_data_or_seg_to_shape.__name__,
        "resampling_fn_seg": resample_data_or_seg_to_shape.__name__,
        "resampling_fn_data_kwargs": rdk,
        "resampling_fn_seg_kwargs": rsk,
        "resampling_fn_probabilities": resample_data_or_seg_to_shape.__name__,
        "resampling_fn_probabilities_kwargs": rpk,
        "architecture": arch,
        "batch_dice": False,
    }
