"""Shared segmentation-experiment code (exp01, exp02, later pipeline.py): load the segmenter once, whole-volume prompted
inference on a scan that is preprocessed once, native-space GT lesion instances with click seeds, the click builders
(noisy / subset / decoy), per-lesion scoring, the eval-manifest reader and the small mask/click file formats.

Non-obvious choices, all deliberate:
- Everything lives in the scan's NATIVE grid (the preprocessed-space sidecars are never used): lesion instances are cc3d
  connectivity-26 components of the binary GT (as `nanounet/data/valset/build.py`), predicted components use
  connectivity 18 (as `nanounet.score`, protocol `longiseg_lesion_v1`); the two are kept as they are on purpose.
- A lesion's click seed is the argmax-EDT voxel of its component (the rule of `nanounet/prompt/centroids.py::_one_case`,
  which guarantees the seed lies inside the lesion), with the EDT taken in mm and the component padded by one voxel so
  the scan border counts as background; at training the grid is ~isotropic, native grids have 3 mm slices.
- The empirical registration-error table is in RESAMPLED voxels (spacing_zyx of the table): a draw is converted to mm and
  then to native voxels of this scan; the size bin comes from the lesion's equivalent-sphere diameter in mm.
- Randomness is a function of (seed, case id, purpose) only (crc32, never Python's salted hash), so any subset of cases,
  any order and any resume gives the same clicks; offsets are drawn once per (lesion, replicate) and only scaled by s
  afterwards, which keeps the exp02 curves nested.
- `predict_case_logits` returns nothing for an empty click list, so the "no click" scenario is a tile placed where a click
  would be with the prompt channels zeroed (the validation manifest's `none_clicked` patch), never an empty call.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
from __future__ import annotations

import json
import os
import zlib
from dataclasses import dataclass
from pathlib import Path

import cc3d
import numpy as np
import SimpleITK as sitk
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree

from experiments.common import SEG_CKPT, SEG_EMA, SEG_MODEL_DIR, abort_if, problem
from nanounet.common import quiet_lightning_runtime
from nanounet.config import load_config
from nanounet.data.patch.error_table import DEFAULT_BACKENDS, DEFAULT_ERROR_TABLE, load_table, sample_offset_vox
from nanounet.data.store.io import SimpleITKIO
from nanounet.data.volume.resampling import set_resample_device
from nanounet.infer.export.volume import native_seg_from_logits
from nanounet.infer.predict.case import MAX_BORDER_EXTRA, predict_case_logits
from nanounet.infer.predict.predictor import load_net_from_ckpt
from nanounet.plan.plans import Plans
from nanounet.plan.prep.case_pp import run_case_npy
from nanounet.score import IOU_HIT, NSD_TOL_MM, dice, iou, nsd

GT_CONNECTIVITY = 26  # GT lesion instances, as nanounet/data/valset/build.py
PRED_CONNECTIVITY = 18  # predicted components, as nanounet.score
DECOY_GUARD_MM = 5.0  # training guard: 5 resampled voxels (~1 mm each) from any lesion
TISSUE_HU = -500.0  # decoys sit on voxels above this (soft tissue, bone; not air or lung)
SCENARIOS = ("S1", "S1_noprompt", "S2", "S3", "S4")
SCENARIOS_BY_ANNOTATION = {"full": ("S1", "S1_noprompt", "S2", "S3", "S4"), "partial": ("S1", "S1_noprompt"), "pseudo": ("S1", "S1_noprompt"),
                           "spheres": ("S1", "S1_noprompt"), "healthy": ("S3", "S4")}
DETECTION_ONLY = ("pseudo", "spheres")  # labels that support "was it found" but not Dice/NSD
TIERS = ("seen-cohort", "outside", "healthy")
OVERLAP_KEYS = ("ours", "nninteractive", "uls_plus")
METHOD_OVERLAP_KEY = {"nanounet": "ours", "nninteractive": "nninteractive", "uls_plus": "uls_plus"}
MANIFEST_SCHEMA = "seg-eval-manifest/1"
CASE_KEYS = ("case_id", "patient_id", "tier", "source", "cancer_type", "annotation", "image", "label", "lesion_label_values", "overlap")
MASK_PAD_VOX = 6  # window padding around a lesion so NSD's erosion never sees the window border
BACKEND_CHOICES = tuple(DEFAULT_BACKENDS)


def rng_for(seed: int, case_id: str, purpose: str, index: int = 0) -> np.random.Generator:
    """Generator that depends only on (seed, case, purpose, index)."""
    return np.random.default_rng([seed, zlib.crc32(case_id.encode()), zlib.crc32(purpose.encode()), index])


@dataclass
class Segmenter:
    net: object
    lm: object
    cfg: object
    pl: object
    cm: object
    dj: dict
    dev: torch.device
    use_tta: bool
    batch_size: int = 8


@dataclass
class Scan:
    pad: torch.Tensor  # (C, Z, Y, X) preprocessed and patch-padded, pinned on cuda
    slicer_revert: tuple
    props: dict
    spacing_zyx: tuple[float, float, float]  # native
    shape_zyx: tuple[int, int, int]  # native
    sitk_stuff: dict  # spacing/origin/direction, to write masks on the scan's grid


def load_segmenter(device: str) -> Segmenter:
    """The chosen segmenter (common.SEG_CKPT, EMA weights) loaded once, same call chain as segtrack/cli/run.py."""
    quiet_lightning_runtime()
    md = str(SEG_MODEL_DIR)
    pl = Plans(join(md, "plans.json"))
    cm, dj, cfg = pl.get_configuration("3d_fullres"), load_json(join(md, "dataset.json")), load_config(join(md, "nano_config.json"))
    set_resample_device(dev := torch.device(device))
    net, lm = load_net_from_ckpt(str(SEG_CKPT), cm, dj, dev, ema=SEG_EMA)
    return Segmenter(net, lm, cfg, pl, cm, dj, dev, use_tta=not cfg.inference.disable_tta_default)


def read_ct(path: str | Path) -> tuple[np.ndarray, dict]:
    """(1, Z, Y, X) float32 CT and its properties (props['spacing'] is native zyx in mm)."""
    return SimpleITKIO().read_images((str(path),))


def prepare_scan(sg: Segmenter, data: np.ndarray, props: dict) -> Scan:
    """Crop, normalise, resample and patch-pad once; every scenario pass on this scan reuses the result."""
    shape, spacing, stuff = tuple(int(s) for s in data.shape[1:]), tuple(float(s) for s in props["spacing"]), props["sitk_stuff"]
    data_pp, _, props = run_case_npy(data, None, props, sg.pl, sg.cm, sg.dj, verbose=False)
    pad, slicer_revert = pad_nd_image(torch.from_numpy(data_pp).float(), tuple(sg.cm.patch_size), "constant", {"value": 0}, True, None)
    return Scan(pad.pin_memory() if sg.dev.type == "cuda" else pad, slicer_revert, props, spacing, shape, stuff)


def segment_points(sg: Segmenter, scan: Scan, points_xyz: list[tuple[float, float, float]], *, encode_prompt: bool = True) -> np.ndarray:
    """Whole-volume prompted inference, tiles only near the clicks (clustered); bool mask on the native grid.
    Same knobs as segtrack: border expansion on, AMP on, TTA per the model config, batch 8."""
    logits, tiles = predict_case_logits(
        net=sg.net, lm=sg.lm, cfg=sg.cfg, pl=sg.pl, cm=sg.cm, dev=sg.dev, pad=scan.pad, slicer_revert=scan.slicer_revert, props=scan.props,
        points_xyz=points_xyz, encode_prompt=encode_prompt, use_tta=sg.use_tta, border_expand=True, max_border_expand_extra=MAX_BORDER_EXTRA,
        batch_size=sg.batch_size, use_amp=True, cluster_margin_frac=0.1, mode="clustered")
    mask = native_seg_from_logits(logits, scan.props, sg.cm, sg.pl, tiles) > 0
    assert mask.shape == scan.shape_zyx, f"native prediction {mask.shape} vs scan {scan.shape_zyx}"
    return mask


def read_labels(path: str | Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))  # held in a variable: never GetArrayViewFromImage(ReadImage(...))
    return sitk.GetArrayFromImage(img)


def read_mask(path: str | Path) -> np.ndarray:
    return read_labels(path) > 0


def write_mask(mask: np.ndarray, sitk_stuff: dict, path: Path) -> None:
    """uint8 NIfTI on the scan's grid, written atomically so a killed run never leaves a half file."""
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.SetSpacing(tuple(sitk_stuff["spacing"]))
    img.SetOrigin(tuple(sitk_stuff["origin"]))
    img.SetDirection(tuple(sitk_stuff["direction"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("part_" + path.name)
    sitk.WriteImage(img, str(tmp), True)
    os.replace(tmp, path)


def size_bin(diam_mm: float) -> tuple[int, str]:
    """Bin of the registration-error table (same rule as error_table._bin_index) and its label."""
    bins = load_table(DEFAULT_ERROR_TABLE)["size_bins_mm"]
    idx = next((i for i, (lo, hi) in enumerate(bins) if lo <= diam_mm < hi), len(bins) - 1)
    lo, hi = bins[idx]
    return idx, (f">{lo:g}" if hi >= 1e8 else f"{lo:g}-{hi:g}")


def read_lesions(label_path: str | Path, values: list[int], spacing_zyx: tuple[float, float, float]) -> tuple[np.ndarray, list[dict]]:
    """Native GT instances (cc3d-26 of the lesion label values) and one dict per lesion: id, seed_zyx (argmax EDT in mm),
    volume, equivalent-sphere size in mm and its size bin, bbox. Every seed lies inside its own component (asserted)."""
    arr = read_labels(label_path)
    inst, n = cc3d.connected_components(np.isin(arr, values).astype(np.uint8), connectivity=GT_CONNECTIVITY, return_N=True)
    stats = cc3d.statistics(inst, no_slice_conversion=False)
    boxes, counts, vox_mm3, lesions = stats["bounding_boxes"], stats["voxel_counts"], float(np.prod(spacing_zyx)), []
    for i in range(1, n + 1):
        sl = boxes[i]
        edt = distance_transform_edt(np.pad(inst[sl] == i, 1), sampling=spacing_zyx)[1:-1, 1:-1, 1:-1]
        seed = np.unravel_index(int(np.argmax(edt)), edt.shape)
        seed = [int(seed[d]) + sl[d].start for d in range(3)]
        assert inst[tuple(seed)] == i, f"{label_path}: seed {seed} of component {i} is not inside it"
        diam = 2.0 * (3.0 * int(counts[i]) * vox_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        lesions.append({"id": i, "seed_zyx": seed, "volume_vox": int(counts[i]), "volume_mm3": int(counts[i]) * vox_mm3, "size_mm": diam,
                        "size_bin": size_bin(diam)[1], "bbox": [int(v) for s_ in sl for v in (s_.start, s_.stop)]})
    return inst, lesions


def draw_offsets(lesions: list[dict], rng: np.random.Generator, backends: tuple[str, ...]) -> list[tuple[float, float, float]]:
    """One empirical registration offset per lesion, in the table's RESAMPLED voxels (zyx), size-matched. Drawn for every
    lesion in id order so a lesion's draw never depends on a lesion cap."""
    tsp = np.asarray(load_table(DEFAULT_ERROR_TABLE)["spacing_zyx"], dtype=float)
    return [sample_offset_vox(l["volume_mm3"] / float(np.prod(tsp)), DEFAULT_ERROR_TABLE, backends, rng) for l in lesions]


def offset_click(lesion: dict, offset_rvox: tuple[float, float, float], s: float, spacing_zyx: tuple[float, float, float],
                 shape_zyx: tuple[int, int, int]) -> tuple[int, int, int]:
    """Seed displaced by s * offset: table voxels -> mm -> native voxels of this scan, rounded and clipped into the volume."""
    tsp = np.asarray(load_table(DEFAULT_ERROR_TABLE)["spacing_zyx"], dtype=float)
    click = np.rint(np.asarray(lesion["seed_zyx"]) + s * np.asarray(offset_rvox) * tsp / np.asarray(spacing_zyx))
    return tuple(int(v) for v in np.clip(click, 0, np.asarray(shape_zyx) - 1))


def offset_mm(click_zyx: tuple[int, int, int], seed_zyx: list[int], spacing_zyx: tuple[float, float, float]) -> tuple[list[float], float]:
    """Effective displacement of the integer click from the seed: (dz, dy, dx) in mm and its magnitude."""
    d = (np.asarray(click_zyx) - np.asarray(seed_zyx)) * np.asarray(spacing_zyx)
    return [float(v) for v in d], float(np.linalg.norm(d))


def pick_subset(ids: list[int], rng: np.random.Generator) -> list[int]:
    """Strict, non-empty subset (k drawn in 1..n-1, as the validation manifest); needs at least 2 ids."""
    assert len(ids) >= 2, f"a strict subset needs >=2 lesions, got {len(ids)}"
    return sorted(int(i) for i in rng.choice(ids, size=int(rng.integers(1, len(ids))), replace=False))


def cap_lesions(ids: list[int], cap: int, rng: np.random.Generator) -> list[int]:
    return ids if cap < 0 or len(ids) <= cap else sorted(int(i) for i in rng.choice(ids, size=cap, replace=False))


def decoy_click(ct_zyx: np.ndarray, inst: np.ndarray, lesions: list[dict], spacing_zyx: tuple[float, float, float],
                rng: np.random.Generator) -> tuple[int, int, int]:
    """One click on tissue (CT above TISSUE_HU) at least DECOY_GUARD_MM from every lesion voxel (KD-tree over lesion
    surface voxels in mm; rejection sampling as nanounet's `_sample_false_pos`, but in the whole scan)."""
    sp = np.asarray(spacing_zyx)
    surf = []
    for l in lesions:
        sl = tuple(slice(l["bbox"][2 * d], l["bbox"][2 * d + 1]) for d in range(3))
        m = np.pad(inst[sl] == l["id"], 1)
        s = np.argwhere((m & ~binary_erosion(m))[1:-1, 1:-1, 1:-1]) + np.array([x.start for x in sl])
        surf.append(s)
    tree = cKDTree(np.concatenate(surf) * sp) if surf else None
    for _ in range(64):
        cand = np.stack([rng.integers(0, d, size=1024) for d in ct_zyx.shape], axis=1)
        ok = ct_zyx[cand[:, 0], cand[:, 1], cand[:, 2]] > TISSUE_HU
        if tree is not None:
            ok &= tree.query(cand * sp, k=1)[0] > DECOY_GUARD_MM
        if ok.any():
            return tuple(int(v) for v in cand[np.argmax(ok)])
    abort_if([problem(f"no decoy position found in a scan of shape {ct_zyx.shape} after 64 x 1024 draws",
                      f"tissue voxels (CT > {TISSUE_HU:g} HU) at least {DECOY_GUARD_MM:g} mm from every lesion",
                      "check the image (empty or all-air scan?) and drop the case from the manifest")])
    raise AssertionError("unreachable: abort_if exits")


def write_clicks(path: Path, clicks: list[tuple[str, tuple[int, int, int]]]) -> None:
    """Click JSON in the pipeline's format, native voxels [x, y, z]: {"points": [{"name", "point"}]}; [] = no click."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"points": [{"name": str(n), "point": [int(x), int(y), int(z)]} for n, (z, y, x) in clicks]}
    tmp = path.with_name("part_" + path.name)
    tmp.write_text(json.dumps(body))
    os.replace(tmp, path)


def read_clicks(path: Path) -> list[tuple[str, tuple[int, int, int]]]:
    return [(str(p["name"]), (int(p["point"][2]), int(p["point"][1]), int(p["point"][0]))) for p in json.loads(path.read_text())["points"]]


def score_lesions(pred: np.ndarray, inst: np.ndarray, lesions: list[dict], spacing_zyx: tuple[float, float, float], *, detection_only: bool = False) -> dict[int, dict]:
    """Per-lesion iou / hit (IoU > IOU_HIT) / Dice / NSD@NSD_TOL_MM, protocol `longiseg_lesion_v1`: the prediction of a lesion is
    the connectivity-18 component with the largest overlap with it (empty if none). Works in windows around each lesion."""
    lab, boxes = None, None
    if pred.any():
        lab = cc3d.connected_components(pred.astype(np.uint8), connectivity=PRED_CONNECTIVITY)
        boxes = cc3d.statistics(lab, no_slice_conversion=False)["bounding_boxes"]
    out = {}
    for l in lesions:
        gw = tuple(slice(l["bbox"][2 * d], l["bbox"][2 * d + 1]) for d in range(3))
        k = 0
        if lab is not None:
            hit = lab[gw][inst[gw] == l["id"]]
            hit = hit[hit != 0]
            k = int(np.bincount(hit.astype(np.int64)).argmax()) if hit.size else 0
        if k == 0:
            out[l["id"]] = {"iou": 0.0, "hit": 0.0, "dice": None if detection_only else 0.0, "nsd": None if detection_only else 0.0, "pred_vox": 0, "gt_vox": l["volume_vox"]}
            continue
        w = tuple(slice(max(0, min(a.start, b.start) - MASK_PAD_VOX), min(n, max(a.stop, b.stop) + MASK_PAD_VOX)) for a, b, n in zip(gw, boxes[k], pred.shape))
        g, p = inst[w] == l["id"], lab[w] == k
        j = iou(g, p)
        out[l["id"]] = {"iou": j, "hit": float(j > IOU_HIT), "dice": None if detection_only else dice(g, p), "nsd": None if detection_only else nsd(g, p, spacing_zyx),
                        "pred_vox": int(p.sum()), "gt_vox": int(g.sum())}
    return out


def fg_stats(pred: np.ndarray, inst: np.ndarray, target: list[dict] | None = None) -> dict:
    """Whole-volume foreground facts: predicted voxels total / inside any GT lesion / outside all GT, and (with `target`, the
    prompted lesions) the foreground Dice against their union and against the union of all GT (the S2 selectivity margin)."""
    n = int(pred.sum())
    in_gt = int((pred & (inst > 0)).sum()) if n else 0
    out = {"fg_vox": n, "fg_vox_in_gt": in_gt, "fg_vox_outside_gt": n - in_gt, "any_fg": float(n > 0), "any_fg_outside_gt": float(n - in_gt > 0)}
    if target is not None:
        union = np.zeros(pred.shape, dtype=bool)
        for l in target:
            w = tuple(slice(l["bbox"][2 * d], l["bbox"][2 * d + 1]) for d in range(3))
            union[w] |= inst[w] == l["id"]
        out["fg_dice_vs_clicked"], out["fg_dice_vs_all"] = dice(union, pred), dice(inst > 0, pred)
        out["fg_dice_margin"] = out["fg_dice_vs_clicked"] - out["fg_dice_vs_all"]
    return out


def manifest_problems(m: dict, path: Path, tiers: tuple[str, ...] = TIERS) -> list[str]:
    """Every schema violation of the eval manifest, as E1 problems (the schema is in the experiments plan, Interfaces block)."""
    fix = "python -m experiments.exp00c_seg_eval_manifest.run (agent B's manifest) or pass --manifest <file with schema seg-eval-manifest/1>"
    if m.get("schema") != MANIFEST_SCHEMA:
        return [problem(f"{path} has schema {m.get('schema')!r}", f"schema {MANIFEST_SCHEMA!r}", fix)]
    out = []
    for c in m.get("cases", []):
        miss = [k for k in CASE_KEYS if k not in c]
        if miss:
            out.append(problem(f"manifest case {c.get('case_id', '?')} lacks keys {miss}", f"keys {list(CASE_KEYS)}", fix))
        elif c["tier"] not in tiers or c["annotation"] not in SCENARIOS_BY_ANNOTATION or set(c["overlap"]) != set(OVERLAP_KEYS) or (c["label"] is None) != (c["annotation"] == "healthy"):
            out.append(problem(f"manifest case {c['case_id']}: tier {c['tier']!r}, annotation {c['annotation']!r}, overlap keys {sorted(c['overlap'])}, label {c['label']!r} are inconsistent",
                               f"tier in {list(tiers)}, annotation in {list(SCENARIOS_BY_ANNOTATION)}, overlap keys {list(OVERLAP_KEYS)}, label null only for healthy", fix))
    return out or ([] if m.get("cases") else [problem(f"{path} holds no cases", "a non-empty cases list", fix)])


def policy_cases(cases: list[dict], methods: list[str], policy: str) -> list[dict]:
    """Cases each method may be scored on: `common` = clean for all three systems (the headline set, same for every method),
    `own` = clean for the method's own system; returns the union over `methods`, in manifest order."""
    if policy == "common":
        return [c for c in cases if all(c["overlap"][k] == "clean" for k in OVERLAP_KEYS)]
    keys = {METHOD_OVERLAP_KEY[m] for m in methods}
    return [c for c in cases if any(c["overlap"][k] == "clean" for k in keys)]


def method_ok(case: dict, method: str, policy: str) -> bool:
    return all(case["overlap"][k] == "clean" for k in OVERLAP_KEYS) if policy == "common" else case["overlap"][METHOD_OVERLAP_KEY[method]] == "clean"


def scenarios_for(case: dict, n_lesions: int) -> tuple[str, ...]:
    """Scenarios a case supports (plan Sec. 6 exp01): full = S1, its prompt-drop twin, S2 (needs >=2 lesions), S3, S4; partial / pseudo /
    spheres = S1 and its twin only; healthy = S3, S4. A case without any lesion keeps only S3 and S4 (healthy-like scan)."""
    sc = SCENARIOS_BY_ANNOTATION[case["annotation"]]
    if n_lesions == 0:
        return tuple(s for s in sc if s in ("S3", "S4"))
    return tuple(s for s in sc if not (s == "S2" and n_lesions < 2))
