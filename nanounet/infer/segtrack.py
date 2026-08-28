"""Predict FU (and BL unless a GT instance mask is given) → linked tracking ids.

Matcher defaults come from tracking.common (v7_complete, EMA, dust_tau=0.125).
Each CT is SimpleITK-read once; XYZ/RAS is reused by the matcher (no second load).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from nanounet.common import cprint
from nanounet.data.io import SimpleITKIO
from nanounet.infer.export import native_seg_from_logits
from nanounet.infer.predict_case import MAX_BORDER_EXTRA, predict_case_logits
from nanounet.infer.predict_io import preprocess_loaded
from nanounet.infer.segtrack_case import SegTrackCase, load_ct, load_instance_zyx, stem_pid_region

DEFAULT_MODEL = Path(
    "/nnunet_data/NanoUNet_results/nanounet/"
    "Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0_h200_instance_1200ep"
)


def _write_mha(vol_zyx: np.ndarray, props: dict, path: Path) -> None:
    SimpleITKIO().write_seg(vol_zyx, str(path), props)


def segment_native(net, lm, cfg, pl, cm, dev, pack, *,
                   use_tta, border_expand=True, max_border_extra=MAX_BORDER_EXTRA, batch_size=8,
                   use_amp=True, cluster_margin_frac=0.1, inference_mode="clustered",
                   no_prompt_encode=False) -> tuple[np.ndarray, dict]:
    pad_cpu, slicer_revert, props, points_xyz, bl_points = pack
    pad = pad_cpu.pin_memory().to(dev, non_blocking=True) if dev.type == "cuda" else pad_cpu.to(dev)
    seg, tiles = predict_case_logits(
        net=net, lm=lm, cfg=cfg, pl=pl, cm=cm, dev=dev,
        pad=pad, slicer_revert=slicer_revert, props=props, points_xyz=points_xyz,
        encode_prompt=not no_prompt_encode, use_tta=use_tta,
        border_expand=border_expand, max_border_expand_extra=max_border_extra,
        batch_size=batch_size, use_amp=use_amp,
        cluster_margin_frac=cluster_margin_frac, mode=inference_mode,
        is_longi=False, bl_present=False, bl_points_xyz=bl_points,
    )
    return native_seg_from_logits(seg, props, cm, pl, tiles), props


def load_case_io(case: SegTrackCase):
    """CPU-only: disk reads. No GPU/preprocess work here -- safe to run in a background thread
    while the GPU is busy on a different case."""
    if case.bl_mask is not None:
        fu = load_ct(case.fu_img)
        bl_zyx, props_bl = load_instance_zyx(case.bl_mask)
        return {"fu": fu, "bl_zyx": bl_zyx, "props_bl": props_bl}
    return {"bl": load_ct(case.bl_img), "fu": load_ct(case.fu_img)}


def run_case(case: SegTrackCase, case_dir: Path, *, net, lm, cfg, pl, cm, dj, dev, matcher,
             decode: str, overwrite: bool, keep_pred: bool, track_ckpt: Path, thresh: float,
             device: str, seg_kw: dict, on_step=None, preloaded: dict | None = None) -> dict:
    from tracking.common import DEPLOYED_DUST_TAU
    from tracking.data.instances import binary_to_instances, load_clicks
    from tracking.data.paint import fu_track_map, paint_fu, write_empty_csv
    from tracking.data.propagate import load_propagated
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
        step("load FU")
        fu_data, fu_props, (ct_fu, aff_fu, sp_fu) = preloaded["fu"] if preloaded else load_ct(case.fu_img)
        pred_bl = None
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(load_ct, case.bl_img)
            bl_zyx, props_bl = (preloaded["bl_zyx"], preloaded["props_bl"]) if preloaded else load_instance_zyx(case.bl_mask)
            step("segment FU")
            pack = preprocess_loaded(fu_data, fu_props, str(case.fu_clicks), pl, cm, dj)
            pred_fu, props_fu = segment_native(net, lm, cfg, pl, cm, dev, pack, **seg_kw)
            _, _, (ct_bl, aff_bl, sp_bl) = fut.result()
        fu_zyx = binary_to_instances(pred_fu, load_clicks(case.fu_clicks))
    else:
        step("segment BL")
        bl_data, bl_raw, (ct_bl, aff_bl, sp_bl) = preloaded["bl"] if preloaded else load_ct(case.bl_img)

        def _fu_pack():
            d, p, trip = preloaded["fu"] if preloaded else load_ct(case.fu_img)
            return preprocess_loaded(d, p, str(case.fu_clicks), pl, cm, dj), trip

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_fu_pack)
            pack_bl = preprocess_loaded(bl_data, bl_raw, str(case.bl_clicks), pl, cm, dj)
            pred_bl, props_bl = segment_native(net, lm, cfg, pl, cm, dev, pack_bl, **seg_kw)
            pack_fu, (ct_fu, aff_fu, sp_fu) = fut.result()
        step("segment FU")
        pred_fu, props_fu = segment_native(net, lm, cfg, pl, cm, dev, pack_fu, **seg_kw)
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
    mk_bl = np.ascontiguousarray(bl_zyx.transpose(2, 1, 0))
    mk_fu = np.ascontiguousarray(fu_zyx.transpose(2, 1, 0))
    if case.bl_mask is not None and mk_bl.shape != ct_bl.shape:
        return {"status": "skip", "n_pairs": 0, "sec": time.perf_counter() - t0, "why": "BL mask grid != BL CT grid"}
    drop_dp = bool(getattr(matcher.hparams, "drop_dp", False))
    prop = case.meta_csv if case.meta_csv is not None else case.fu_clicks
    _, region = stem_pid_region(case.stem)
    img_id = region if case.meta_csv is not None else None
    if not drop_dp:
        bl_ids = [int(x) for x in np.unique(bl_zyx) if int(x) != 0]
        got, _ = load_propagated(prop, bl_ids, img_id=img_id)
        drop = sorted(set(bl_ids) - set(got))
        if drop:
            cprint(f"[dim]drop {case.stem}  BL ids {drop} (not in this FU volume)[/dim]")
    r = track(
        case.bl_img, case.bl_img, case.fu_img, case.fu_img,
        None if drop_dp else prop, track_ckpt,
        decode=decode, device=device, matcher=matcher, thresh=thresh,
        sinkhorn_tau=DEPLOYED_DUST_TAU, use_ema=True,
        types_csv=case.types_csv, img_id=img_id,
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
