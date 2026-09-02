"""Native lesion metrics: volume Dice + per-lesion DSC/NSD/LDR.

Empty GT instance → NaN (dropped). Empty pred + nonempty GT → 0.
Pred CC = click voxel if on FG, else the cc3d-18 component with max overlap vs that GT instance.
"""

import csv
import json
import os

import cc3d
import numpy as np
import SimpleITK as sitk
from rich.panel import Panel
from scipy.ndimage import binary_erosion, distance_transform_edt

from nanounet.common import cprint

IOU_HIT = 0.1
NSD_TOL_MM = 1.0
_DOC = "docs/steps/predict.md"

def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    den = float(gt.sum() + pred.sum())
    return 1.0 if den == 0.0 else 2.0 * float(np.logical_and(gt, pred).sum()) / den

def iou(gt: np.ndarray, pred: np.ndarray) -> float:
    if not gt.any():
        return float("nan")
    inter = float(np.logical_and(gt, pred).sum())
    union = float(gt.sum() + pred.sum()) - inter
    return 0.0 if union == 0.0 else inter / union

def nsd(gt: np.ndarray, pred: np.ndarray, spacing_zyx: tuple[float, float, float], tol: float = NSD_TOL_MM) -> float:
    if not gt.any():
        return float("nan")
    if not pred.any():
        return 0.0
    m = 1 + max(int(np.ceil(tol / s)) for s in spacing_zyx)
    sl = tuple(slice(max(0, int(c.min()) - m), min(d, int(c.max()) + 1 + m)) for c, d in zip(np.where(gt | pred), gt.shape))
    gt, pred = gt[sl], pred[sl]
    def surf(mask):
        s = mask & ~binary_erosion(mask)
        return s if s.any() else mask
    sg, sp = surf(gt), surf(pred)
    d_g = distance_transform_edt(~sp, sampling=spacing_zyx)[sg]
    d_p = distance_transform_edt(~sg, sampling=spacing_zyx)[sp]
    n = int(d_g.size + d_p.size)
    return 1.0 if n == 0 else float(((d_g <= tol).sum() + (d_p <= tol).sum()) / n)

def check_gt_dir(gt_dir: str, cases: list, end: str) -> None:
    missing = [cid for cid, *_ in cases if not os.path.isfile(os.path.join(gt_dir, cid + end))]
    if missing:
        raise FileNotFoundError(
            f"No GT for: {', '.join(missing)}.\nExpected instance-labeled '{{stem}}{end}' in '{gt_dir}' (same stems as -i).\n"
            f"Fix: pass --gt-dir /nnunet_data/Longitudinal-CT/targetsTrFU   (see {_DOC})"
        )


def _clicks(path: str) -> dict[int, tuple[int, int, int]]:
    with open(path, encoding="utf-8") as f:
        pts = json.load(f).get("points")
    if not isinstance(pts, list):
        raise KeyError(
            f"'points' missing or not a list in '{path}'.\nExpected {{'points': [{{'name': '<id>', 'point': [x,y,z]}}, ...]}}.\n"
            f"Fix: sibling <case>.json next to each scan   (see {_DOC})"
        )
    out: dict[int, tuple[int, int, int]] = {}
    for item in pts:
        raw = item.get("name") if isinstance(item, dict) else None
        try:
            lid = int(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"Click in '{path}' has missing or non-integer name: {item!r}.\nExpected points[].name to be the lesion_id integer.\n"
                f"Fix: use sibling JSON from inputsTrFU   (see {_DOC})"
            ) from None
        p = item["point"]
        out.setdefault(lid, (int(round(p[2])), int(round(p[1])), int(round(p[0]))))
    return out


def score_case(case_id: str, pred_path: str, gt_path: str, clicks_json: str) -> dict:
    pimg, gimg = sitk.ReadImage(pred_path), sitk.ReadImage(gt_path)
    pred, gt = sitk.GetArrayFromImage(pimg) > 0, np.asarray(sitk.GetArrayFromImage(gimg))
    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch for '{case_id}': pred {pred.shape} vs GT {gt.shape}.\nExpected native GT on the same grid as the prediction.\n"
            f"Fix: pass --gt-dir with instance masks matching -i   (see {_DOC})"
        )
    clicks = _clicks(clicks_json)
    if float(gt.max()) == 1 and len(clicks) >= 2:
        raise ValueError(
            f"GT at '{gt_path}' looks binary (max={gt.max()}).\n"
            f"Expected instance-labeled masks (voxel value = lesion_id), e.g. targetsTrFU.\n"
            f"Fix: pass --gt-dir /nnunet_data/Longitudinal-CT/targetsTrFU   (see {_DOC})"
        )
    osh, sp = pred.shape, pimg.GetSpacing()
    spacing = (float(sp[2]), float(sp[1]), float(sp[0]))
    fg = np.where(pred | (gt > 0))
    z0 = y0 = x0 = 0
    if fg[0].size:
        z0, y0, x0 = int(fg[0].min()), int(fg[1].min()), int(fg[2].min())
        sl = (slice(z0, int(fg[0].max()) + 1), slice(y0, int(fg[1].max()) + 1), slice(x0, int(fg[2].max()) + 1))
        pred, gt = pred[sl], gt[sl]
    sh = pred.shape
    lab = cc3d.connected_components(pred.astype(np.uint8), connectivity=18)
    lesions, n_skip = [], 0
    for lid, (z, y, x) in clicks.items():
        # round-then-clip, matching the encoding convention in prompt/coords.py:53-56 —
        # sub-voxel float noise from the click pipeline can land right on a boundary.
        z = min(max(z, 0), osh[0] - 1)
        y = min(max(y, 0), osh[1] - 1)
        x = min(max(x, 0), osh[2] - 1)
        gt_i = gt == lid
        if not gt_i.any():
            n_skip += 1
            continue
        hit = lab[gt_i]
        hit = hit[hit != 0]
        pred_i = (lab == np.bincount(hit).argmax()) if hit.size else np.zeros(sh, dtype=bool)
        j = iou(gt_i, pred_i)
        lesions.append({"id": lid, "dsc": dice(gt_i, pred_i), "nsd": nsd(gt_i, pred_i, spacing),
                        "iou": j, "ldr": float(j > IOU_HIT)})
    nan = float("nan")
    return {
        "case_id": case_id, "volume_dice": dice(gt > 0, pred), "lesions": lesions,
        "dsc": float(np.nanmean([x["dsc"] for x in lesions])) if lesions else nan,
        "nsd": float(np.nanmean([x["nsd"] for x in lesions])) if lesions else nan,
        "ldr": float(np.nanmean([x["ldr"] for x in lesions])) if lesions else nan,
        "n": len(lesions), "n_skip": n_skip,
    }


def _f(v) -> str:
    return "—" if isinstance(v, float) and v != v else f"{v:.3f}"

def _dsc_cell(v) -> str:
    s = _f(v)
    return s if s == "—" else (f"[red]{s}[/red]" if v < 0.10 else f"[green]{s}[/green]" if v >= 0.70 else s)


def _agg(rows: list[dict]) -> dict:
    all_d = [L["dsc"] for r in rows for L in r["lesions"]]
    return {
        "n_cases": len(rows), "n_lesions": sum(r["n"] for r in rows),
        "n_skipped_empty_gt": sum(r["n_skip"] for r in rows),
        "volume_dice_mean": float(np.mean([r["volume_dice"] for r in rows])),
        "volume_dice_median": float(np.median([r["volume_dice"] for r in rows])),
        "dsc_case_mean": float(np.nanmean([r["dsc"] for r in rows])),
        "dsc_case_median": float(np.nanmedian([r["dsc"] for r in rows])),
        "nsd_case_mean": float(np.nanmean([r["nsd"] for r in rows])),
        "ldr_case_mean": float(np.nanmean([r["ldr"] for r in rows])),
        "dsc_lesion_macro": float(np.nanmean(all_d)) if all_d else float("nan"),
        "dsc_lesion_median": float(np.nanmedian(all_d)) if all_d else float("nan"),
    }


def report_case(r: dict) -> None:
    ldr = f"[dim]{_f(r['ldr'])}[/dim]" if r["n"] == 0 else _f(r["ldr"])
    cprint(f"         n={r['n']}  Dice vol {_f(r['volume_dice'])}  DSC {_dsc_cell(r['dsc'])}  NSD {_f(r['nsd'])}  LDR {ldr}")


def report(rows: list[dict]) -> None:
    a = _agg(rows)
    cprint(Panel(
        f"Dice vol     {_f(a['volume_dice_mean'])}  (median {_f(a['volume_dice_median'])})   case-mean whole-volume (not comparable to LongiSeg)\n"
        f"DSC          {_f(a['dsc_case_mean'])}  (median {_f(a['dsc_case_median'])})   case-mean per-lesion  ← vs LongiSeg verified ~0.737\n"
        f"NSD          {_f(a['nsd_case_mean'])}    1 mm surface\n"
        f"LDR          {_f(a['ldr_case_mean'])}    IoU > {IOU_HIT}\n"
        f"lesions      {a['n_lesions']} scored, {a['n_skipped_empty_gt']} empty-GT skipped",
        title="LongiSeg lesion metrics", border_style="cyan"))
    cprint(f"[dim]lesion-macro DSC {_f(a['dsc_lesion_macro'])}  (median {_f(a['dsc_lesion_median'])})  (every lesion equal; headline is case-mean)[/dim]")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    return None if isinstance(obj, float) and obj != obj else obj


def write(rows: list[dict], path: str) -> None:
    stem = os.path.splitext(path)[0]
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump({"protocol": "longiseg_lesion_v1", "nsd_tol_mm": NSD_TOL_MM, "ldr_iou": IOU_HIT,
                   "overall": _jsonable(_agg(rows)), "cases": _jsonable(rows)}, f, indent=2)
    with open(stem + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "lesion_id", "dsc", "nsd", "iou", "ldr", "volume_dice"])
        for r in rows:
            for L in r["lesions"]:
                w.writerow([r["case_id"], L["id"], L["dsc"], L["nsd"], L["iou"], L["ldr"], r["volume_dice"]])
    cprint(f"[dim]wrote {stem}.json  {stem}.csv\nnext: {stem}.json[/dim]")
