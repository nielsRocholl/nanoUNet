"""ROI / prompt JSON config → frozen dataclasses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Tuple, cast

from nanounet.data.patch.error_table import parse_propagated


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
    # Absent/empty => uniform case draw, exactly. See nanounet/data/patch/cohorts.py.
    cohorts: Mapping[str, float] = field(default_factory=dict)
    # False (default): a missing <case>_weights.json falls back to uniform per-centroid sampling.
    # True: that same absence raises instead. See nanounet/data/patch/sampling.py.
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
        raise KeyError(
            f"Missing required config key {key!r}.\n"
            f"Expected every ROI-prompt config JSON to define it (see nanounet/configs/default.json for a working example).\n"
            f"Fix: add \"{key}\": ... to your config JSON   (see nanounet/docs/reference/config.md)"
        )
    return d[key]


def _load_prop(d: dict | None) -> PropagatedConfig:
    kw = parse_propagated(d)
    kw["mode"] = cast(Literal["gaussian", "empirical"], kw["mode"])
    return PropagatedConfig(**kw)


def _load_sampling(d: dict) -> SamplingConfig:
    fgp = float(_require(d, "fg_patch_prob"))
    if not 0.0 <= fgp <= 1.0:
        raise ValueError(
            f"sampling.fg_patch_prob={fgp} is outside [0, 1].\n"
            f"Expected a probability in [0, 1] (see nanounet/configs/default.json for a working example).\n"
            f"Fix: set sampling.fg_patch_prob to a value in [0, 1]   (see nanounet/docs/reference/config.md)"
        )
    cm = _require(d, "click_modes")
    assert isinstance(cm, dict)
    p = float(cm["pos"])
    dr = float(cm["drop"])
    if p < 0 or p > 1 or dr < 0 or dr > 1:
        raise ValueError(
            f"sampling.click_modes.pos={p} and/or drop={dr} are outside [0, 1].\n"
            f"Expected both to be probabilities in [0, 1] (see nanounet/configs/default.json).\n"
            f"Fix: set sampling.click_modes.pos and .drop to values in [0, 1]   (see nanounet/docs/reference/config.md)"
        )
    if abs(p + dr - 1.0) > 1e-5:
        raise ValueError(
            f"sampling.click_modes.pos={p} and drop={dr} do not sum to 1.\n"
            f"Expected sampling.click_modes.pos + .drop to sum to exactly 1 (see nanounet/configs/default.json).\n"
            f"Fix: adjust sampling.click_modes.pos/.drop so they sum to 1   (see nanounet/docs/reference/config.md)"
        )
    fp_prob = float(d.get("false_pos_probability", 1.0))
    if fp_prob < 0 or fp_prob > 1:
        raise ValueError(
            f"sampling.false_pos_probability={fp_prob} is outside [0, 1].\n"
            f"Expected a probability in [0, 1] (see nanounet/configs/default.json).\n"
            f"Fix: set sampling.false_pos_probability to a value in [0, 1]   (see nanounet/docs/reference/config.md)"
        )
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
        raise ValueError(
            f"prompt.encoding={enc!r} is not a supported encoding.\n"
            f"Expected one of: binary, edt (see nanounet/configs/default.json).\n"
            f"Fix: set prompt.encoding to \"binary\" or \"edt\"   (see nanounet/docs/reference/config.md)"
        )
    sc = float(d.get("prompt_intensity_scale", 1.0))
    if sc <= 0 or sc > 1:
        raise ValueError(
            f"prompt.prompt_intensity_scale={sc} is outside (0, 1].\n"
            f"Expected a value greater than 0 and at most 1 (see nanounet/configs/default.json).\n"
            f"Fix: set prompt.prompt_intensity_scale to a value in (0, 1]   (see nanounet/docs/reference/config.md)"
        )
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
        raise ValueError(
            f"validation.no_lesion_frac={f} is outside [0, 1].\n"
            f"Expected a fraction in [0, 1] (see nanounet/configs/default.json).\n"
            f"Fix: set validation.no_lesion_frac to a value in [0, 1]   (see nanounet/docs/reference/config.md)"
        )
    return ValidationConfig(no_lesion_frac=f)


def load_config(path: str | Path) -> RoiPromptConfig:
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(
            f"Config file {p} does not parse to a JSON object (dict).\n"
            f"Expected a top-level object, as in nanounet/configs/default.json.\n"
            f"Fix: wrap the config in a top-level JSON object   (see nanounet/docs/reference/config.md)"
        )
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
