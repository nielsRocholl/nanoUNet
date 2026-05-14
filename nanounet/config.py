"""ROI / prompt JSON config → frozen dataclasses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Tuple, cast


@dataclass(frozen=True)
class LargeLesionConfig:
    K: Tuple[int, int]
    K_min: int
    K_max: int
    max_extra: int


@dataclass(frozen=True)
class PropagatedConfig:
    sigma_per_axis: Tuple[float, float, float]
    max_vox: float


@dataclass(frozen=True)
class SamplingConfig:
    mode_probs: Tuple[float, float, float, float]
    n_spur: Tuple[int, int]
    n_neg: Tuple[int, int]
    large_lesion: LargeLesionConfig
    propagated: PropagatedConfig


@dataclass(frozen=True)
class PromptConfig:
    point_radius_vox: int
    encoding: Literal["binary", "edt"]
    validation_use_prompt: bool
    prompt_intensity_scale: float


@dataclass(frozen=True)
class InferenceConfig:
    tile_step_size: float
    disable_tta_default: bool


@dataclass(frozen=True)
class RoiPromptConfig:
    prompt: PromptConfig
    sampling: SamplingConfig
    inference: InferenceConfig


def _require(d: dict, key: str) -> object:
    if key not in d:
        raise KeyError(key)
    return d[key]


def _parse_int_range(val: object, key: str) -> Tuple[int, int]:
    if isinstance(val, int):
        return (val, val)
    if isinstance(val, (list, tuple)) and len(val) == 2:
        lo, hi = int(val[0]), int(val[1])
        if lo > hi:
            raise ValueError(f"{key}: min>max {val}")
        return (lo, hi)
    raise ValueError(f"{key}: int or [min,max]")


def _load_large(d: dict) -> LargeLesionConfig:
    return LargeLesionConfig(
        K=_parse_int_range(_require(d, "K"), "K"),
        K_min=int(_require(d, "K_min")),
        K_max=int(_require(d, "K_max")),
        max_extra=int(_require(d, "max_extra")),
    )


def _load_prop(d: dict | None) -> PropagatedConfig:
    if not isinstance(d, dict):
        return PropagatedConfig(sigma_per_axis=(2.75, 5.19, 5.40), max_vox=34.0)
    sg = d.get("sigma_per_axis", (2.75, 5.19, 5.40))
    assert isinstance(sg, (list, tuple)) and len(sg) == 3
    return PropagatedConfig(
        sigma_per_axis=tuple(float(x) for x in sg),
        max_vox=float(d.get("max_vox", 34.0)),
    )


def _load_sampling(d: dict) -> SamplingConfig:
    mp = _require(d, "mode_probs")
    assert isinstance(mp, (list, tuple)) and len(mp) == 4
    probs = tuple(float(x) for x in mp)
    if abs(sum(probs) - 1.0) > 1e-5:
        raise ValueError("mode_probs must sum to 1")
    ll = _require(d, "large_lesion")
    assert isinstance(ll, dict)
    return SamplingConfig(
        mode_probs=probs,
        n_spur=_parse_int_range(_require(d, "n_spur"), "n_spur"),
        n_neg=_parse_int_range(_require(d, "n_neg"), "n_neg"),
        large_lesion=_load_large(ll),
        propagated=_load_prop(d.get("propagated")),
    )


def _load_prompt(d: dict) -> PromptConfig:
    enc = str(_require(d, "encoding"))
    if enc not in ("binary", "edt"):
        raise ValueError(enc)
    sc = float(d.get("prompt_intensity_scale", 1.0))
    if sc <= 0 or sc > 1:
        raise ValueError("prompt_intensity_scale in (0,1]")
    return PromptConfig(
        point_radius_vox=int(_require(d, "point_radius_vox")),
        encoding=cast(Literal["binary", "edt"], enc),
        validation_use_prompt=bool(d.get("validation_use_prompt", False)),
        prompt_intensity_scale=sc,
    )


def _load_inf(d: dict | None) -> InferenceConfig:
    if not isinstance(d, dict):
        return InferenceConfig(0.5, False)
    return InferenceConfig(
        float(d.get("tile_step_size", 0.5)),
        bool(d.get("disable_tta_default", False)),
    )


def load_config(path: str | Path) -> RoiPromptConfig:
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError("config root must be dict")
    pr = _require(d, "prompt")
    sa = _require(d, "sampling")
    assert isinstance(pr, dict) and isinstance(sa, dict)
    return RoiPromptConfig(
        prompt=_load_prompt(pr),
        sampling=_load_sampling(sa),
        inference=_load_inf(d.get("inference")),
    )


def save_config(cfg: RoiPromptConfig, path: str | Path) -> None:
    def ser(d: object) -> object:
        if hasattr(d, "__dataclass_fields__"):
            return {k: ser(v) for k, v in asdict(d).items()}
        if isinstance(d, tuple):
            return list(d)
        return d

    Path(path).write_text(json.dumps(ser(cfg), indent=2), encoding="utf-8")
