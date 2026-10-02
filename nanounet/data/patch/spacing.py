"""Bind the voxel spacing that turns registration-error table offsets (mm) into voxels.

Median-mode plans have ONE spacing for the whole dataset (bound once from the plans). Z-only plans
resample each case differently, so the spacing is the case's own `spacing_after_resampling` and is
bound per case where the properties are in hand (build_patch, valset case_info). Z-only plans leave
the dataset-level value unbound so a case without the key fails loudly (draw_propagated_offset)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from nanounet.config import RoiPromptConfig


def bind_roi_spacing(cfg: "RoiPromptConfig", spacing_zyx: Tuple[float, float, float]) -> "RoiPromptConfig":
    """cfg with sampling.propagated.data_spacing_zyx = the plans' `3d_fullres.spacing` (array-axis
    order), so empirical offsets scale mm -> data voxels. Call once where the plans are known."""
    sp = tuple(float(x) for x in spacing_zyx)
    prop = replace(cfg.sampling.propagated, data_spacing_zyx=sp)
    return replace(cfg, sampling=replace(cfg.sampling, propagated=prop))


def bind_plan_spacing(cfg: "RoiPromptConfig", cm) -> "RoiPromptConfig":
    """Dataset-level spacing for median-mode plans; z-only plans leave it unbound because every case
    carries its own `spacing_after_resampling` (build_patch / valset case_info bind it per case)."""
    if cm.configuration.get("spacing_mode", "median") == "z_only":
        return cfg
    return bind_roi_spacing(cfg, cm.spacing)


def bind_case_spacing(cfg: "RoiPromptConfig", properties: dict) -> "RoiPromptConfig":
    sp = properties.get("spacing_after_resampling")
    return cfg if sp is None else bind_roi_spacing(cfg, sp)


def case_spacing(case, prop_cfg) -> tuple:
    """Valset case's spacing: its own spacing_after_resampling, else the dataset-level one."""
    sp = case.spacing or prop_cfg.data_spacing_zyx
    if sp is None:
        raise ValueError(
            f"No voxel spacing for case {case.cid}: no spacing_after_resampling and no dataset spacing bound by the plans.\n"
            f"Expected a case written by the current nanounet_preprocess.\n"
            f"Fix: nanounet_preprocess -d <id> --resume (see nanounet/docs/steps/preprocess.md)"
        )
    return sp
