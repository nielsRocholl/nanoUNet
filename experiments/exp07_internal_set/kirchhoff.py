# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
"""Kirchhoff et al. (LongiSeg, prompted longitudinal segmentation) as the comparison system of exp07, exp08 and exp09.

Runs LongiSeg's own `predict_case` in-process (same Python environment as everything else, `common.ensure_longiseg_env()`), exactly as
`/nnunet_data/LongiSeg/scripts/predict_nanounet_testset.py` does: one call per follow-up scan with the BL image, the annotated BL instance
mask, the BL click points and the FU click points; the BL scan is picked per FU scan by click-id overlap; folds 0-4, the function's own
defaults (no TTA). The output is one instance mask per FU scan whose voxel value is the prompt (lesion) id.

Identity is INHERITED from the prompt (plan Sec. 6 exp07): prompt i names BL lesion i, so the FU lesion segmented from prompt i is its
link; the annotated FU lesion it overlaps (IoU > 0.1, `scoring.match_nodes`, same rule as our pipeline) is the link's FU endpoint, and a
segmentation overlapping no annotated lesion is an extra node (negative id, a false-positive edge). A prompt that segments nothing links
nothing (correct for a disappeared lesion). The method has no way to declare a lesion new: a lesion that no prompt reaches is not a node
at all, and one that a prompt does reach is linked to that prompt, so the class `new` is not defined for it and run.py reports it as null.
Non-obvious: `predict_case` is hard-wired to CUDA, so `--device` only steers our pipeline; it loads the five fold checkpoints on every call
(part of its per-case cost, about 2.2 s per lesion plus loading).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from experiments.common import ensure_longiseg_env
from experiments.scoring import PairCase, match_nodes

LONGISEG_FOLDS = ("fold_0", "fold_1", "fold_2", "fold_3", "fold_4")
MODEL_FILES = ("plans.json", "dataset.json")


def load_clicks(path: Path) -> dict[int, tuple]:
    """Click JSON `{"points": [{"name": id, "point": [x, y, z]}]}` -> {lesion id: point} (voxel, native grid), as LongiSeg's script reads it."""
    return {int(p["name"]): tuple(p["point"]) for p in json.loads(Path(path).read_text())["points"]}


def model_problems(model: Path) -> list[tuple[str, str]]:
    """(what, path) for every file of a LongiSeg model dir that is missing (plans, dataset, one checkpoint per fold)."""
    need = [model / f for f in MODEL_FILES] + [model / f / "checkpoint_final.pth" for f in LONGISEG_FOLDS]
    return [(f"LongiSeg model file {p.relative_to(model)}", str(p)) for p in need if not p.is_file()]


def pick_baseline(bl_dir: Path, pid: str, fu_ids: set[int]) -> str | None:
    """BL stem of the patient whose click ids overlap the FU click ids most (first wins a tie): the script's rule, because a patient's BL can
    enumerate more lesions than one FU scan covers and `predict_case` only predicts ids present in both click sets."""
    best, best_n = None, 0
    for nii in sorted(bl_dir.glob(f"{pid}_*.nii.gz")):
        stem = nii.name[: -len(".nii.gz")]
        if (bl_dir / f"{stem}.json").is_file() and (n := len(set(load_clicks(bl_dir / f"{stem}.json")) & fu_ids)) > best_n:
            best, best_n = stem, n
    return best


def predict(model: Path, pid: str, fu_img: Path, fu_clicks: Path, bl_img_dir: Path, bl_mask_dir: Path, out_nii: Path) -> dict:
    """One FU scan through LongiSeg; writes the instance mask to `out_nii`. Returns {bl_stem, prompt_ids (BL and FU), fu_click_ids, t_kirchhoff, predict_status}.
    `predict_status` is `no_output` when `predict_case` found no lesion with a valid BL+FU pair (nothing is written, everything counts as missed)."""
    ensure_longiseg_env()
    import torch
    from longiseg.inference.autopet_inference import predict_case  # only importable after ensure_longiseg_env

    fu = load_clicks(fu_clicks)
    bl_stem = pick_baseline(bl_img_dir, pid, set(fu))
    if bl_stem is None:
        raise FileNotFoundError(f"no baseline scan of {pid} whose click ids overlap the follow-up click ids {sorted(fu)[:8]}. "
                                f"Fix: put <pid>_<idx>.nii.gz and .json for the baseline scan in {bl_img_dir}")
    bl = load_clicks(bl_img_dir / f"{bl_stem}.json")
    t0 = time.perf_counter()
    try:
        result = predict_case({"primary_bl_image_path": str(bl_img_dir / f"{bl_stem}.nii.gz"), "primary_fu_image_path": str(fu_img),
                               "primary_bl_mask_path": str(bl_mask_dir / f"{bl_stem}.nii.gz"), "primary_bl_clickpoints": bl, "primary_fu_clickpoints": fu},
                              Path(model))
    finally:
        torch.cuda.empty_cache()
    rec = {"bl_stem": bl_stem, "prompt_ids": sorted(set(bl) & set(fu)), "fu_click_ids": sorted(fu), "t_kirchhoff": time.perf_counter() - t0, "predict_status": "ok"}
    if "primary" not in result:
        return {**rec, "predict_status": "no_output"}
    out_nii.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(result["primary"], str(out_nii))
    return rec


def read_prediction(path: Path) -> np.ndarray:
    """Instance prediction (Z, Y, X) int32, voxel value = prompt id."""
    img = sitk.ReadImage(str(path))  # keep the image alive while the array is copied (never a view of a temporary)
    return np.rint(sitk.GetArrayFromImage(img)).astype(np.int32)


def identity(case: PairCase, pred_inst: np.ndarray | None, gt_fu_inst: np.ndarray, prompt_ids: list[int]) -> tuple[PairCase, dict]:
    """The case with found_bl, found_fu and pred_links filled from a Kirchhoff prediction (see the module docstring), plus the prompt ->
    lesion assignment {prompt id: {"n_vox": int, "annotated_fu": id or None}}. `pred_inst` None = `predict_case` produced nothing."""
    from dataclasses import replace

    hit = match_nodes(pred_inst, gt_fu_inst) if pred_inst is not None else {}
    present = {int(i): int(n) for i, n in zip(*np.unique(pred_inst[pred_inst > 0], return_counts=True))} if pred_inst is not None else {}
    bl_annotated = set(case.bl_ids)
    found_bl = {i for i in prompt_ids if i in bl_annotated}
    links, found_fu, assign = set(), set(), {}
    for i in sorted(prompt_ids):
        if i not in present:
            assign[i] = {"n_vox": 0, "annotated_fu": None}
            continue
        j = hit.get(i)
        assign[i] = {"n_vox": present[i], "annotated_fu": j}
        if j is not None:
            found_fu.add(j)
        # a prompt that names no annotated BL lesion has an extra BL endpoint (negative id), so its link is always a false positive
        links.add((i if i in bl_annotated else -(i + 1), j if j is not None else -(i + 1)))
    return replace(case, found_bl=found_bl, found_fu=found_fu, pred_links=links), assign
