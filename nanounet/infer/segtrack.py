"""Predict FU (and BL unless a GT instance mask is given) → linked tracking ids."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from nanounet.data.io import SimpleITKIO
from nanounet.infer.export import native_seg_from_logits
from nanounet.infer.predict_case import MAX_BORDER_EXTRA, predict_case_logits
from nanounet.infer.predict_io import preprocess_case
from nanounet.infer.segtrack_case import SegTrackCase, load_instance_zyx

DEFAULT_MODEL = Path(
    "/nnunet_data/NanoUNet_results/nanounet/"
    "Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0_h200_instance_1200ep"
)
DEFAULT_TRACK = Path("/nnunet_data/lesion_tracking/runs/h60_r9/best.ckpt")


def _write_mha(vol_zyx: np.ndarray, props: dict, path: Path) -> None:
    SimpleITKIO().write_seg(vol_zyx, str(path), props)


def segment_native(net, lm, cfg, pl, cm, dj, dev, scan: Path, clicks: Path, *,
                   use_tta, border_expand=True, max_border_extra=MAX_BORDER_EXTRA, batch_size=8,
                   use_amp=True, cluster_margin_frac=0.1, inference_mode="clustered",
                   no_prompt_encode=False, pack=None) -> tuple[np.ndarray, dict]:
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
    return native_seg_from_logits(logits, props, cm, pl, dj, tiles), props


def run_case(case: SegTrackCase, case_dir: Path, *, net, lm, cfg, pl, cm, dj, dev, matcher,
             decode: str, overwrite: bool, keep_pred: bool, track_ckpt: Path, thresh: float,
             device: str, seg_kw: dict, on_step=None) -> dict:
    from tracking.data.graph import _load_vol
    from tracking.data.instances import binary_to_instances, load_clicks
    from tracking.data.paint import fu_track_map, paint_fu, write_empty_csv
    from tracking.infer import track, write_match_csv

    def step(s: str) -> None:
        if on_step:
            on_step(s)

    t0 = time.perf_counter()
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    csv_path = case_dir / "matches.csv"
    if csv_path.is_file() and not overwrite:
        return {"status": "skip", "n_pairs": 0, "sec": 0.0}
    assert (case.bl_mask is None) != (case.bl_clicks is None)

    if case.bl_mask is not None:
        step("load BL mask")
        bl_zyx, props_bl = load_instance_zyx(case.bl_mask)
        pred_bl = None
        step("segment FU")
        pred_fu, props_fu = segment_native(net, lm, cfg, pl, cm, dj, dev, case.fu_img, case.fu_clicks, **seg_kw)
        fu_zyx = binary_to_instances(pred_fu, load_clicks(case.fu_clicks))
    else:
        step("segment BL")
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(preprocess_case, str(case.fu_img), str(case.fu_clicks), pl, cm, dj, None, None)
            pred_bl, props_bl = segment_native(net, lm, cfg, pl, cm, dj, dev, case.bl_img, case.bl_clicks, **seg_kw)
            pack_fu = fut.result()
        step("segment FU")
        pred_fu, props_fu = segment_native(
            net, lm, cfg, pl, cm, dj, dev, case.fu_img, case.fu_clicks, pack=pack_fu, **seg_kw,
        )
        bl_zyx = binary_to_instances(pred_bl, load_clicks(case.bl_clicks))
        fu_zyx = binary_to_instances(pred_fu, load_clicks(case.fu_clicks))
    has_bl, has_fu = bool(np.any(bl_zyx)), bool(np.any(fu_zyx))

    def emit_preds() -> None:
        if not keep_pred:
            return
        if pred_bl is not None:
            _write_mha(pred_bl, props_bl, case_dir / "pred_bl.mha")
        _write_mha(pred_fu, props_fu, case_dir / "pred_fu.mha")

    if not has_bl or not has_fu:
        bl_out = bl_zyx if has_bl else np.zeros(bl_zyx.shape, dtype=np.int32)
        fu_out = fu_zyx if has_fu else np.zeros(pred_fu.shape, dtype=np.int32)
        _write_mha(bl_out, props_bl, case_dir / "bl.mha")
        _write_mha(fu_out, props_fu, case_dir / "fu.mha")
        write_empty_csv(csv_path)
        emit_preds()
        return {"status": "empty", "n_pairs": 0, "sec": time.perf_counter() - t0}

    step("track")
    ct_bl, aff_bl, sp_bl = _load_vol(case.bl_img)
    ct_fu, aff_fu, sp_fu = _load_vol(case.fu_img)
    mk_bl = np.ascontiguousarray(bl_zyx.transpose(2, 1, 0))
    mk_fu = np.ascontiguousarray(fu_zyx.transpose(2, 1, 0))
    if case.bl_mask is not None and mk_bl.shape != ct_bl.shape:
        raise SystemExit(
            f"BL mask grid {tuple(mk_bl.shape)} != BL CT grid {tuple(ct_bl.shape)} for {case.stem}.\n"
            f"Expected a native baseline instance mask on the same grid as {case.bl_img}.\n"
            f"Fix: --bl-mask /nnunet_data/Longitudinal-CT/targetsTrBL/{case.stem}.nii.gz  (see docs/steps/track.md)"
        )
    r = track(
        case.bl_img, case.bl_img, case.fu_img, case.fu_img,
        case.fu_clicks, track_ckpt,
        decode=decode, device=device, matcher=matcher, thresh=thresh,
        types_csv=case.types_csv,
        volumes=(ct_bl, aff_bl, sp_bl, mk_bl, ct_fu, aff_fu, sp_fu, mk_fu),
    )
    m = fu_track_map(
        list(map(int, r.bl_ids)), list(map(int, r.fu_ids)),
        [(int(r.bl_ids[i]), int(r.fu_ids[j])) for i, j in r.pairs],
    )
    _write_mha(bl_zyx, props_bl, case_dir / "bl.mha")
    _write_mha(paint_fu(fu_zyx, m), props_fu, case_dir / "fu.mha")
    write_match_csv(csv_path, r)
    emit_preds()
    return {"status": "ok", "n_pairs": len(r.pairs), "sec": time.perf_counter() - t0}
