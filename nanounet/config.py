"""ROI / prompt JSON config → frozen dataclasses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Tuple, cast

from nanounet.data.error_table import parse_propagated


@dataclass(frozen=True)
class PropagatedConfig:
    mode: Literal["gaussian", "empirical"]
    error_table: str
    backends: Tuple[str, ...]
    sigma_per_axis: Tuple[float, float, float]
    max_vox: float


@dataclass(frozen=True)
class ClickModeConfig:
    pos: float
    drop: float


@dataclass(frozen=True)
class SamplingConfig:
    fg_patch_prob: float
    click_modes: ClickModeConfig
    false_pos_probability: float
    propagated: PropagatedConfig
    instance_targets: bool = False
    # Absent/empty => uniform case draw, exactly. See nanounet/data/cohorts.py.
    cohorts: Mapping[str, float] = field(default_factory=dict)
    # False (default): a missing <case>_weights.json falls back to uniform per-centroid sampling.
    # True: that same absence raises instead. See nanounet/data/sampling.py.
    require_weights: bool = False


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
class ValidationConfig:
    no_lesion_frac: float


@dataclass(frozen=True)
class RoiPromptConfig:
    prompt: PromptConfig
    sampling: SamplingConfig
    inference: InferenceConfig
    validation: ValidationConfig


def _require(d: dict, key: str) -> object:
    if key not in d:
        raise KeyError(key)
    return d[key]


def _load_prop(d: dict | None) -> PropagatedConfig:
    kw = parse_propagated(d)
    kw["mode"] = cast(Literal["gaussian", "empirical"], kw["mode"])
    return PropagatedConfig(**kw)


def _load_sampling(d: dict) -> SamplingConfig:
    fgp = float(_require(d, "fg_patch_prob"))
    if not 0.0 <= fgp <= 1.0:
        raise ValueError("fg_patch_prob must be in [0, 1]")
    cm = _require(d, "click_modes")
    assert isinstance(cm, dict)
    p = float(cm["pos"])
    dr = float(cm["drop"])
    if p < 0 or p > 1 or dr < 0 or dr > 1:
        raise ValueError("click_modes pos and drop must be in [0, 1]")
    if abs(p + dr - 1.0) > 1e-5:
        raise ValueError("click_modes.pos + click_modes.drop must sum to 1")
    fp_prob = float(d.get("false_pos_probability", 1.0))
    if fp_prob < 0 or fp_prob > 1:
        raise ValueError("false_pos_probability must be in [0, 1]")
    return SamplingConfig(
        fg_patch_prob=fgp,
        click_modes=ClickModeConfig(pos=p, drop=dr),
        false_pos_probability=fp_prob,
        propagated=_load_prop(d.get("propagated")),
        instance_targets=bool(d.get("instance_targets", False)),
        cohorts={str(k): float(v) for k, v in (d.get("cohorts") or {}).items()},
        require_weights=bool(d.get("require_weights", False)),
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


def _load_validation(d: dict | None) -> ValidationConfig:
    f = float(d.get("no_lesion_frac", 0.3)) if isinstance(d, dict) else 0.3
    if not 0.0 <= f <= 1.0:
        raise ValueError("validation.no_lesion_frac must be in [0, 1]")
    return ValidationConfig(no_lesion_frac=f)


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
        validation=_load_validation(d.get("validation")),
    )


def save_config(cfg: RoiPromptConfig, path: str | Path) -> None:
    def ser(d: object) -> object:
        if hasattr(d, "__dataclass_fields__"):
            return {k: ser(v) for k, v in asdict(d).items()}
        if isinstance(d, tuple):
            return list(d)
        return d

    Path(path).write_text(json.dumps(ser(cfg), indent=2), encoding="utf-8")
