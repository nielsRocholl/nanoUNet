"""Native CT+clicks → predict both timepoints → instance masks with shared tracking ids."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from nanounet.common import results_dir
from nanounet.infer.export import export_prediction_from_logits
from nanounet.infer.predict_case import MAX_BORDER_EXTRA, predict_case_logits
from nanounet.infer.predict_io import preprocess_case

DEFAULT_MODEL = Path(
    "/nnunet_data/NanoUNet_results/nanounet/"
    "Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0_h200_instance_1200ep"
)
DEFAULT_TRACK = Path("/nnunet_data/lesion_tracking/runs/h60_r9/best.ckpt")
END = ".nii.gz"


@dataclass
class SegTrackCase:
    stem: str
    bl_img: Path
    bl_clicks: Path
    fu_img: Path
    fu_clicks: Path
    types_csv: Path | None = None


def resolve_ckpt_path(cli: str | None, env_key: str, default: Path) -> tuple[Path, str]:
    if cli:
        return Path(cli), "cli"
    v = os.environ.get(env_key)
    if v:
        return Path(v), "env"
    return default, "default"


def resolve_out(stem: str, *, fu_dir_name: str | None, out: Path | None, single: bool) -> Path:
    if out is not None:
        return Path(out) if single else Path(out) / stem
    root = Path(results_dir()) / "segtrack"
    return root / "single" / stem if single else root / str(fu_dir_name) / stem


def _stems(folder: Path) -> dict[str, tuple[Path, Path]]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"No input folder at {folder}.\n"
            f"Expected a folder of {{stem}}{END} + sibling {{stem}}.json.\n"
            f"Fix: --bl-dir / --fu-dir like inputsTrBL / inputsTrFU  (see docs/steps/track.md)"
        )
    out, missing = {}, []
    for p in sorted(folder.glob(f"*{END}")):
        stem = p.name[: -len(END)]
        js = folder / f"{stem}.json"
        out[stem] = (p, js)
        if not js.is_file():
            missing.append(stem)
    if missing:
        raise SystemExit(
            f"missing points JSON for: {', '.join(missing[:12])}.\n"
            f"Expected sibling <case>.json next to each scan in {folder}.\n"
            f"Fix: add the JSON  (see docs/steps/track.md)"
        )
    if not out:
        raise SystemExit(
            f"No {END} scans in {folder}.\n"
            f"Expected sibling .nii.gz + .json like inputsTrFU.\n"
            f"Fix: pass --bl-dir / --fu-dir  (see docs/steps/track.md)"
        )
    return out


def pair_folder(bl_dir: Path, fu_dir: Path) -> list[SegTrackCase]:
    bl, fu = _stems(Path(bl_dir)), _stems(Path(fu_dir))
    only_bl, only_fu = sorted(set(bl) - set(fu)), sorted(set(fu) - set(bl))
    if only_bl or only_fu:
        raise SystemExit(
            "BL/FU folders do not share the same case names.\n"
            f"--bl-dir has {len(only_bl)} stems not in --fu-dir (e.g. {', '.join(only_bl[:12]) or 'none'}).\n"
            f"--fu-dir has {len(only_fu)} stems not in --bl-dir (e.g. {', '.join(only_fu[:12]) or 'none'}).\n"
            "Fix: pass matching inputsTrBL and inputsTrFU, or --patients-csv to select a subset\n"
            "(see docs/steps/track.md)"
        )
    return [SegTrackCase(s, *bl[s], *fu[s]) for s in sorted(bl)]


def segment_native(net, lm, cfg, pl, cm, dj, dev, scan: Path, clicks: Path, out_nii: Path, *,
                   use_tta, border_expand=True, max_border_extra=MAX_BORDER_EXTRA, batch_size=8,
                   use_amp=True, cluster_margin_frac=0.1, inference_mode="clustered",
                   no_prompt_encode=False, pack=None) -> None:
    if pack is None:
        pack = preprocess_case(str(scan), str(clicks), pl, cm, dj, None, None)
    pad_cpu, slicer_revert, props, points_xyz, bl_points = pack
    pad = pad_cpu.pin_memory().to(dev, non_blocking=True) if dev.type == "cuda" else pad_cpu.to(dev)
    logits, tiles = predict_case_logits(
        net=net, lm=lm, cfg=cfg, pl=pl, cm=cm, dev=dev,
        pad=pad, slicer_revert=slicer_revert, props=props, points_xyz=points_xyz,
        encode_prompt=not no_prompt_encode, use_tta=use_tta,
        border_expand=border_expand, max_border_expand_extra=max_border_extra,
        batch_size=batch_size, use_amp=use_amp,
        cluster_margin_frac=cluster_margin_frac, mode=inference_mode,
        is_longi=False, bl_present=False, bl_points_xyz=bl_points,
    )
    out_trunc = str(out_nii)
    end = dj["file_ending"]
    if out_trunc.endswith(end):
        out_trunc = out_trunc[: -len(end)]
    export_prediction_from_logits(logits, props, cm, pl, dj, out_trunc, tiles)


def run_case(case: SegTrackCase, case_dir: Path, *, net, lm, cfg, pl, cm, dj, dev, matcher,
             decode: str, overwrite: bool, keep_pred: bool, track_ckpt: Path, thresh: float,
             device: str, seg_kw: dict, on_step=None) -> dict:
    from tracking.data.instances import instances_from_nifti
    from tracking.data.paint import copy_mask, fu_track_map, write_case_masks, write_empty_csv
    from tracking.infer import mask_has_lesions, track, write_match_csv

    def step(s: str) -> None:
        if on_step:
            on_step(s)

    t0 = time.perf_counter()
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    csv_path = case_dir / "matches.csv"
    if csv_path.is_file() and not overwrite:
        return {"status": "skip", "n_pairs": 0, "sec": 0.0}
    tmp = Path(tempfile.mkdtemp())
    pred_bl, pred_fu = tmp / "pred_bl.nii.gz", tmp / "pred_fu.nii.gz"
    try:
        step("segment BL")
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(preprocess_case, str(case.fu_img), str(case.fu_clicks), pl, cm, dj, None, None)
            segment_native(net, lm, cfg, pl, cm, dj, dev, case.bl_img, case.bl_clicks, pred_bl, **seg_kw)
            pack_fu = fut.result()
        step("segment FU")
        segment_native(net, lm, cfg, pl, cm, dj, dev, case.fu_img, case.fu_clicks, pred_fu, pack=pack_fu, **seg_kw)
        bl_inst = instances_from_nifti(pred_bl, case.bl_clicks, tmp / "bl_inst.nii.gz")
        fu_inst = instances_from_nifti(pred_fu, case.fu_clicks, tmp / "fu_inst.nii.gz")
        has_bl, has_fu = mask_has_lesions(bl_inst), mask_has_lesions(fu_inst)
        if not has_bl or not has_fu:
            copy_mask(bl_inst, case_dir / "bl.nii.gz", zero=not has_bl)
            copy_mask(fu_inst, case_dir / "fu.nii.gz", zero=not has_fu)
            write_empty_csv(csv_path)
            if keep_pred:
                shutil.copy2(pred_bl, case_dir / "pred_bl.nii.gz")
                shutil.copy2(pred_fu, case_dir / "pred_fu.nii.gz")
            return {"status": "empty", "n_pairs": 0, "sec": time.perf_counter() - t0}
        step("track")
        r = track(
            case.bl_img, bl_inst, case.fu_img, fu_inst, case.fu_clicks, track_ckpt,
            decode=decode, device=device, matcher=matcher, thresh=thresh, types_csv=case.types_csv,
        )
        m = fu_track_map(
            list(map(int, r.bl_ids)), list(map(int, r.fu_ids)),
            [(int(r.bl_ids[i]), int(r.fu_ids[j])) for i, j in r.pairs],
        )
        write_case_masks(bl_inst, fu_inst, case_dir, m)
        write_match_csv(csv_path, r)
        if keep_pred:
            shutil.copy2(pred_bl, case_dir / "pred_bl.nii.gz")
            shutil.copy2(pred_fu, case_dir / "pred_fu.nii.gz")
        return {"status": "ok", "n_pairs": len(r.pairs), "sec": time.perf_counter() - t0}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
