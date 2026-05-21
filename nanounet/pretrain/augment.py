"""MAE pretrain spatial aug: rotation + scale + mirror (nnssl BaseMAE recipe)."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert2DTo3DTransform, Convert3DTo2DTransform

from nanounet.common import ANISO_THRESHOLD


def _rotation_for_da(patch_size: list[int]) -> tuple[tuple[float, float], bool]:
    do_dummy = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
    if do_dummy:
        rot = (-np.pi, np.pi)
    else:
        rot = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
    return rot, do_dummy


def train_spatial_tf(patch_size: Union[np.ndarray, Tuple[int, ...]]) -> BasicTransform:
    ps = [int(x) for x in patch_size]
    rot, do_dummy = _rotation_for_da(ps)
    transforms = []
    if do_dummy:
        transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = ps[1:]
    else:
        patch_size_spatial = ps
    transforms.append(
        SpatialTransform(
            patch_size_spatial,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0,
            p_rotation=0.2,
            rotation=rot,
            p_scaling=0.2,
            scaling=(0.7, 1.4),
            p_synchronize_scaling_across_axes=1,
            bg_style_seg_sampling=False,
        )
    )
    if do_dummy:
        transforms.append(Convert2DTo3DTransform())
    transforms.append(MirrorTransform(allowed_axes=(0, 1, 2)))
    return ComposeTransforms(transforms)


def apply_spatial_tf(tf: BasicTransform, patch: np.ndarray) -> np.ndarray:
    im = torch.from_numpy(patch).float()
    se = torch.zeros((1, *patch.shape[1:]), dtype=torch.short)
    with torch.no_grad():
        o = tf(image=im, segmentation=se)
    return np.asarray(o["image"], dtype=np.float32)
