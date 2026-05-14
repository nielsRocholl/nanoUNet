"""ZScore / CT / none / rescale normalization (names stored in ``plans.json``)."""

from __future__ import annotations

from typing import Type

import numpy as np
from numpy import number


class ZScoreNormalization:
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def __init__(self, use_mask_for_norm: bool | None, intensityproperties: dict, target_dtype: Type[number] = np.float32):
        self.use_mask_for_norm = use_mask_for_norm
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    def run(self, image: np.ndarray, seg: np.ndarray | None = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)
        if self.use_mask_for_norm:
            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / max(std, 1e-8)
        else:
            image -= image.mean()
            image /= max(image.std(), 1e-8)
        return image


class CTNormalization:
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def __init__(self, use_mask_for_norm: bool | None, intensityproperties: dict, target_dtype: Type[number] = np.float32):
        assert intensityproperties is not None
        self.use_mask_for_norm = use_mask_for_norm
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    def run(self, image: np.ndarray, seg: np.ndarray | None = None) -> np.ndarray:
        ip = self.intensityproperties
        image = image.astype(self.target_dtype, copy=False)
        np.clip(image, ip["percentile_00_5"], ip["percentile_99_5"], out=image)
        image -= ip["mean"]
        image /= max(ip["std"], 1e-8)
        return image


class NoNormalization:
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def __init__(self, use_mask_for_norm: bool | None, intensityproperties: dict, target_dtype: Type[number] = np.float32):
        self.use_mask_for_norm = use_mask_for_norm
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    def run(self, image: np.ndarray, seg: np.ndarray | None = None) -> np.ndarray:
        return image.astype(self.target_dtype, copy=False)


class RescaleTo01Normalization:
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def __init__(self, use_mask_for_norm: bool | None, intensityproperties: dict, target_dtype: Type[number] = np.float32):
        self.use_mask_for_norm = use_mask_for_norm
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    def run(self, image: np.ndarray, seg: np.ndarray | None = None) -> np.ndarray:
        image = image.astype(self.target_dtype, copy=False)
        image -= image.min()
        image /= np.clip(image.max(), a_min=1e-8, a_max=None)
        return image


class RGBTo01Normalization:
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def __init__(self, use_mask_for_norm: bool | None, intensityproperties: dict, target_dtype: Type[number] = np.float32):
        self.use_mask_for_norm = use_mask_for_norm
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    def run(self, image: np.ndarray, seg: np.ndarray | None = None) -> np.ndarray:
        assert image.min() >= 0 and image.max() <= 255
        image = image.astype(self.target_dtype, copy=False)
        image /= 255.0
        return image


_BY_NAME = {
    "zscorenormalization": ZScoreNormalization,
    "ctnormalization": CTNormalization,
    "nonormalization": NoNormalization,
    "rescaleto01normalization": RescaleTo01Normalization,
    "rgbto01normalization": RGBTo01Normalization,
}

_CH_MAP = {
    "ct": CTNormalization,
    "nonorm": NoNormalization,
    "zscore": ZScoreNormalization,
    "rescale_to_0_1": RescaleTo01Normalization,
    "rgb_to_0_1": RGBTo01Normalization,
}


def normalization_class_from_plan_name(scheme_name: str) -> type:
    k = scheme_name.lower().split(".")[-1]
    c = _BY_NAME.get(k)
    if c is None:
        raise RuntimeError(scheme_name)
    return c


def normalization_class_for_channel(channel_name: str) -> type:
    return _CH_MAP.get(channel_name.casefold(), ZScoreNormalization)
