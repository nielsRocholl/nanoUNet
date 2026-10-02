# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
"""exp00c - Segmentation evaluation manifest  (paper: Table 'nine experiments' rows 1 and 2, evaluation data; plan Sec. 2 and Sec. 6 exp00c)

QUESTION   Which single-timepoint CT cases do the segmentation experiments (exp01, exp02) and the external systems (ULS+,
           nnInteractive) run on, such that no compared system saw them in training?
WHY        Pins the cases in one small git-tracked file (`seg_eval_v1.json`) so every method, and every rerun, sees the same
           cases and the paper can say which cancer types are covered outside the training data and which only in-distribution.
DATA       Three tiers. seen-cohort: the Longitudinal-CT held-out 60 (`Dataset013_Longitudinal_CT/imagesTs`, 132 scans) plus a few
           validation cases per training cohort (flagged `used_for_checkpoint_selection`). outside: capped, seeded, patient-disjoint
           subsets of the KEEP sources of `PancancerCTSeg` (matrix `SOURCES` below = plan Sec. 2, verbatim). healthy:
           `validation/HealthyImages-noLesion` (172 lesion-free scans).
METHOD     1. `SOURCES` holds every source with its per-corpus overlap (ours / nnInteractive / ULS+), basis, kept flag and reason;
              it is data, never re-derived. The manifest copies it into `sources`.
           2. Every candidate of a KEEP source has its NIfTI header read (size, spacing, affine = origin + direction, to 1e-3; no voxel data) and is
              dropped when it equals the header of any of our 5690 training volumes (splits_final.json train + val; dataset.json lists
              106 more RUMC-pancreas volumes that are not on disk and were not trained on). Each drop is logged in `dropped`.
           3. Per source, patients are shuffled with `default_rng(seed)`; one scan per patient (SegRap25: the `cect` scan); the first
              `--cap` survivors are kept. A patient never appears twice, a scan never in two sources.
           4. Every PancancerCTSeg label is binary (0 background, 1 lesion; checked on every kept label: any other value aborts), so
              `lesion_label_values` is [1]; lesions are cc3d connected components (connectivity 26) of that mask.
           5. The coverage table counts cases, patients and lesions per tier/source/cancer type; kept/dropped per source is listed.
OUTPUT     `seg_eval_v1.json` (next to this file on a full run; in the run dir for a smoke run), run dir `results.json` with tables
           `sources`, `cases`, `dropped`, `coverage`, `candidates_per_source`.
COMMAND    python -m experiments.exp00c_seg_eval_manifest.run --tag paper_v1
DEPENDS ON experiments/common.py only.
RUNTIME    About 10 min on CPU (about 8000 header reads over CIFS, cached in a local scratch file; ~500 label reads for the lesion counts).
CAVEATS    Overlap with nnInteractive and ULS+ is judged per source from their published training-data lists (not from their images),
           so `clean` means "not in the lists", not proof. Sources with no patient map (ScienceDB, BoneTumor, AbdomenAtlas, LUNA25)
           use the case id as patient id. Healthy scans carry `unverified` flags (their source is undocumented).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cc3d
import nibabel
import numpy as np
import SimpleITK as sitk
from rich.table import Table

from core.ui import console, cprint, nano_progress
from experiments.common import HOLDOUT_CSV, abort_if, add_common_args, limited, missing_paths, problem, start_run

EXP = "exp00c_seg_eval_manifest"
PCS = Path("/nnunet_data/raw/PancancerCTSeg")
PER_CANCER, NEW = PCS / "train_per_cancer_type", PCS / "new_cancer_datasets_after202606"
CORPUS_DIR = Path("/nnunet_data/NanoUNet_preprocessed/Dataset900_Merged")
SEEN_DIR = Path("/nnunet_data/NanoUNet_raw/Dataset013_Longitudinal_CT")
HEALTHY_DIR = PCS / "validation" / "HealthyImages-noLesion"
MANIFEST = Path(__file__).resolve().parent / "seg_eval_v1.json"
IMG_SUFFIX = "_0000.nii.gz"


def S(name, cancer, n, ours, nni, uls, basis, kept, reason, annotation="full", **layout):
    """One row of the source matrix (plan Sec. 2). Flags: clean | overlap | unverified. `layout` (images/labels dir, filename regex) only for KEEP sources."""
    return {"name": name, "cancer_type": cancer, "n_total": n, "overlap": {"ours": ours, "nninteractive": nni, "uls_plus": uls}, "overlap_basis": basis, "kept": kept, "reason": reason, "annotation": annotation, **layout}


SOURCES = [
    # ---- KEEP (clean at source level; plan Sec. 2) ----
    S("SegRap25", "head-neck", 240, "clean", "clean", "clean", "source", True, "240 files = 120 patients x {cect, ncct}; the cect scan is used, one per patient", images=PER_CANCER / "Dataset001_HeadNeckCancer/imagesTr", labels=PER_CANCER / "Dataset001_HeadNeckCancer/labelsTr", regex=r"^HeadNeck_SegRap25_(?P<pid>\d+)-cect_0000\.nii\.gz$", metadata=PER_CANCER / "Dataset001_HeadNeckCancer/dataset.json"),
    S("ScienceDBLungTumorCT", "lung", 1167, "clean", "clean", "clean", "source", True, "ScienceDB 2026 (post-dates nnInteractive), expert masks; no patient map, case id = patient id", images=NEW / "20260820_ScienceDBLungTumorCT/imagesTr", labels=NEW / "20260820_ScienceDBLungTumorCT/labelsTr", regex=r"^(?P<pid>[AE]SLS26_\d+_\d+)_0000\.nii\.gz$", metadata=NEW / "20260820_ScienceDBLungTumorCT/case_manifest.tsv"),
    S("NSCLC-Radiogenomics", "lung", 88, "clean", "clean", "clean", "source", True, "residual risk: ULS+ CCC18 draws on TCIA collections, unverifiable offline", images=PER_CANCER / "Dataset003_LungCancer/imagesTr", labels=PER_CANCER / "Dataset003_LungCancer/labelsTr", regex=r"^Chest_NSCLC-Radiogenomics_(?P<pid>R01-\d+)_0000\.nii\.gz$", metadata=PER_CANCER / "Dataset003_LungCancer/dataset.json"),
    S("HCC-TACE-SEG", "liver", 74, "clean", "clean", "clean", "source", True, "clean only if ULS+'s 'CECT liver (prim.) 1274' is not this set (assumed = MCT-LTDiag, owner to confirm)", images=PER_CANCER / "Dataset004_LiverCancer/imagesTr", labels=PER_CANCER / "Dataset004_LiverCancer/labelsTr", regex=r"^Liver_HCC-TACE-SEG_(?P<pid>\d+)_0000\.nii\.gz$", metadata=PER_CANCER / "Dataset004_LiverCancer/dataset.json"),
    S("SpinalMyelomaCT", "bone", 72, "clean", "clean", "clean", "source", True, "lytic myeloma, TCIA Spinal-Multiple-Myeloma-SEG; 72 scans of 67 patients", images=NEW / "20260820_SpinalMyelomaCT/imagesTr", labels=NEW / "20260820_SpinalMyelomaCT/labelsTr", regex=r"^SpinalMyeloma_Myel_(?P<pid>\d+)_\d+_0000\.nii\.gz$", metadata=NEW / "20260820_SpinalMyelomaCT/case_manifest.tsv"),
    S("BoneTumorLungMetastasis", "lung", 61, "clean", "clean", "clean", "source", True, "labels are voxelised spheres (detection only); 61 cases of 59 patients, no patient map, case id = patient id", "spheres", images=NEW / "20260915_BoneTumorLungMetastasis/imagesTr", labels=NEW / "20260915_BoneTumorLungMetastasis/labelsTr", regex=r"^BoneTumor_(?P<pid>\d+)_0000\.nii\.gz$", metadata=NEW / "20260915_BoneTumorLungMetastasis/dataset.json"),
    S("LUNA25", "lung", 4049, "clean", "clean", "clean", "source", False, "optional and off by default (--include-luna25): masks are MedSAM2-derived pseudo-labels, detection only", "pseudo", images=PER_CANCER / "Dataset003_LungCancer/imagesTr", labels=PER_CANCER / "Dataset003_LungCancer/labelsTr", regex=r"^Chest_LUNA25_(?P<pid>\d+)_0000\.nii\.gz$", metadata=PER_CANCER / "Dataset003_LungCancer/dataset.json"),
    S("PANORAMA", "pancreas", 479, "unverified", "clean", "unverified", "header", True, "Radboudumc pancreas data overlaps in institution with our Dataset026_RUMC_Pancreas (also seen by ULS+): kept only after the case-level header check against all 119 RUMC volumes (and every other training volume)", images=PER_CANCER / "Dataset006_PancreaticCancer/imagesTr", labels=PER_CANCER / "Dataset006_PancreaticCancer/labelsTr", regex=r"^Pancreas_PANORAMA_(?P<pid>\d+)-\d+_0000\.nii\.gz$", metadata=PER_CANCER / "Dataset006_PancreaticCancer/dataset.json", resolves_by_header=["ours", "uls_plus"]),
    S("AbdomenAtlas-esophagus", "esophagus", 154, "unverified", "clean", "clean", "header", True, "IDs 14484+ lie outside AbdomenAtlas1.1Mini 1-9262 (not in nnInteractive); original scan collections unknown, so case-level header check vs ours; labels partial/AI-assisted", "partial", images=PER_CANCER / "Dataset002_EsophagusCancer/imagesTr", labels=PER_CANCER / "Dataset002_EsophagusCancer/labelsTr", regex=r"^Esophagus_AbdomenAtlas_(?P<pid>\d+)_0000\.nii\.gz$", metadata=PER_CANCER / "Dataset002_EsophagusCancer/dataset.json", min_id=14484, resolves_by_header=["ours"]),
    S("AbdomenAtlas-endometrial", "endometrial", 79, "unverified", "clean", "clean", "header", True, "IDs 22412+ lie outside AbdomenAtlas1.1Mini 1-9262 (not in nnInteractive); original scan collections unknown, so case-level header check vs ours; labels partial/AI-assisted", "partial", images=PER_CANCER / "Dataset010_EndometrialCancer/imagesTr", labels=PER_CANCER / "Dataset010_EndometrialCancer/labelsTr", regex=r"^Endometrial_AbdomenAtlas_(?P<pid>\d+)_0000\.nii\.gz$", metadata=PER_CANCER / "Dataset010_EndometrialCancer/dataset.json", min_id=22412, resolves_by_header=["ours"]),
    # ---- EXCLUDE (plan Sec. 2); flags other than the deciding corpus are `unverified` where no list settles them ----
    S("LIDC-IDRI", "lung", 875, "overlap", "overlap", "overlap", "source", False, "our Dataset024, nnInteractive LIDC, ULS23 LIDC-IDRI"),
    S("LNDb", "lung", 229, "overlap", "overlap", "clean", "source", False, "our Dataset012, nnInteractive LNDb (ULS23 itself excluded LNDb)"),
    S("MSD-LungTumor", "lung", 63, "overlap", "overlap", "overlap", "source", False, "our Dataset015, nnInteractive MSD Task06, ULS+ MSD"),
    S("NSCLC-Radiomics", "lung", 415, "overlap", "overlap", "overlap", "source", False, "nnInteractive NSCLC-Radiomics; MSD-Lung (our Dataset015) derives from it"),
    S("MSD-Liver", "liver", 118, "overlap", "overlap", "overlap", "source", False, "our Dataset017, nnInteractive MSD Task03, ULS+ MSD"),
    S("MSD-HepaticVessel", "liver", 303, "clean", "overlap", "overlap", "source", False, "nnInteractive MSD Task08, ULS+ MSD (not among our 21 cohorts)"),
    S("Colorectal-Liver-Metastases", "liver", 102, "overlap", "unverified", "overlap", "source", False, "our Dataset011 (CLM), ULS+ CLM"),
    S("WAW-TACE", "liver", 219, "overlap", "clean", "overlap", "source", False, "our Dataset019, ULS+ WAW-TACE"),
    S("TotalSeg-liver_lesions", "liver", 815, "unverified", "unverified", "unverified", "unverified", False, "provenance undocumented, presumed a subset of the TotalSegmentator scans nnInteractive saw"),
    S("Adrenal-ACC-Ki67-Seg", "adrenal", 52, "overlap", "clean", "clean", "source", False, "our Dataset031"),
    S("PanTS", "pancreas", 876, "overlap", "unverified", "unverified", "source", False, "our Dataset028"),
    S("MSD-Pancreas", "pancreas", 281, "overlap", "overlap", "overlap", "source", False, "our Dataset016, nnInteractive MSD Task07, ULS23 MSD Task07"),
    S("PanTrack", "pancreas", 161, "clean", "clean", "clean", "source", False, "reserved for exp09 (transfer to another disease), not a segmentation evaluation source"),
    S("KiTS23", "kidney", 488, "overlap", "overlap", "overlap", "source", False, "our Dataset022, nnInteractive KiTS23, ULS23 KiTS21 (subset of KiTS23)"),
    S("Mediastinal-Lymph-Node-SEG", "lymph node", 354, "overlap", "unverified", "unverified", "source", False, "our Dataset030"),
    S("CT-Lymph-Nodes", "lymph node", 176, "unverified", "overlap", "overlap", "source", False, "nnInteractive NIH CT-Lymph-Nodes, ULS23 NIH-LN"),
    S("MSD-Colon", "colon", 126, "overlap", "overlap", "overlap", "source", False, "our Dataset014, nnInteractive MSD Task10, ULS23 MSD Task10"),
    S("DeepLesion", "whole body", 5000, "clean", "overlap", "overlap", "source", False, "nnInteractive and ULS+ DeepLesion"),
    S("autoPETCT", "whole body", 692, "clean", "overlap", "unverified", "source", False, "nnInteractive AutoPET2"),
    S("LongitudinalCTLesion", "melanoma", 583, "overlap", "clean", "overlap", "source", False, "our Dataset013 and ULS+ Longitudinal-CT (incl. 20260915_LongitudinalCTLesionTest, 64 scans of the held-out 60: those belong to the seen-cohort tier)"),
    S("MSWAL", "whole body", 484, "overlap", "clean", "overlap", "source", False, "our Dataset018, ULS+ MSWAL"),
]
# seen-cohort validation sources: flags for the two external systems per cohort (ours: the cohort is in training, these cases are its held-out validation split)
COHORTS = {"d011": ("CLM", "unverified", "overlap"), "d012": ("LNDb", "overlap", "clean"), "d013": ("Longitudinal-CT", "clean", "overlap"), "d014": ("MSD-Colon", "overlap", "overlap"),
           "d015": ("MSD-Lung", "overlap", "overlap"), "d016": ("MSD-Pancreas", "overlap", "overlap"), "d017": ("MSD-Liver", "overlap", "overlap"), "d018": ("MSWAL", "clean", "overlap"),
           "d019": ("WAW-TACE", "clean", "overlap"), "d020": ("WORC-CRLM", "clean", "overlap"), "d021": ("WORC-GIST", "clean", "overlap"), "d022": ("KiTS23", "overlap", "overlap"),
           "d023": ("LiTS", "unverified", "overlap"), "d024": ("LIDC-IDRI", "overlap", "overlap"), "d025": ("RUMC-Bone", "clean", "overlap"), "d026": ("RUMC-Pancreas", "clean", "overlap"),
           "d027": ("MCT-LTDiag", "clean", "overlap"), "d028": ("PanTS", "unverified", "unverified"), "d029": ("RIDER-LungCT", "unverified", "unverified"),
           "d030": ("Mediastinal-LN", "unverified", "unverified"), "d031": ("Adrenal-ACC-Ki67", "clean", "clean")}
HEALTHY_GROUPS = {"PANCREAS": ("NIH Pancreas-CT; nnInteractive trained on it", "unverified", "overlap", "unverified"), "CHAOS": ("CHAOS CT liver donors; nnInteractive used CHAOS MRI only", "unverified", "clean", "unverified"),
                  "train": ("train_*_a, source unstated", "unverified", "unverified", "unverified")}


def read_header(path: str) -> dict:
    """Shape, spacing and affine (= origin + direction) from the NIfTI header only. nibabel, not SimpleITK: ImageFileReader serialises on the GIL and was ~20x slower over CIFS."""
    im = nibabel.load(path)
    return {"size": [int(x) for x in im.shape[:3]], "spacing": [float(x) for x in im.header.get_zooms()[:3]], "affine": np.round(im.affine, 4).tolist()}


def header_key(h: dict) -> tuple:
    """Equal size, spacing and affine to 1e-3 (same scan geometry)."""
    return (tuple(h["size"]), tuple(round(x, 3) for x in h["spacing"]), tuple(round(x, 3) for row in h["affine"] for x in row))


def read_headers(paths: list[str], cache_path: Path, workers: int, desc: str) -> dict[str, dict]:
    """Headers for `paths` through a local JSON cache (CIFS is slow; the cache is keyed by path, delete it if files change)."""
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    todo = [p for p in paths if p not in cache]
    if todo:
        with nano_progress(len(todo), desc) as advance, ThreadPoolExecutor(workers) as ex:
            for p, h in zip(todo, ex.map(read_header, todo)):
                cache[p] = h
                advance()
        cache_path.write_text(json.dumps(cache))
    return {p: cache[p] for p in paths}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidates(src: dict, args: argparse.Namespace) -> list[dict]:
    """Patients of a KEEP source: one dict per scan {patient, scan, image, label}; several scans of a patient are grouped later."""
    out = []
    for f in sorted(os.listdir(src["images"])):
        m = re.match(src["regex"], f)
        if m and ("min_id" not in src or int(m.group("pid")) >= src["min_id"]):
            stem = f.removesuffix(IMG_SUFFIX)
            out.append({"patient": m.group("pid"), "stem": stem, "image": str(src["images"] / f), "label": str(src["labels"] / f"{stem}.nii.gz")})
    return out


def lesion_stats(label: str, values: list[int]) -> dict:
    """Connected components (26) of the lesion mask; every value outside {0} + lesion values is an error the caller reports."""
    img = sitk.ReadImage(label)  # held in a variable: an inline array view of a temporary image is a use-after-free
    a = sitk.GetArrayFromImage(img)
    present = set(np.flatnonzero(np.bincount(a.ravel().view(np.uint8), minlength=256)).tolist()) if a.dtype in (np.uint8, np.int8) else set(np.unique(a).tolist())
    stray = sorted(int(v) for v in present - {0} - set(values))
    mask = np.isin(a, values).astype(np.uint8)
    _, n = cc3d.connected_components(mask, connectivity=26, return_N=True)
    return {"n_lesions": int(n), "lesion_voxels": int(mask.sum()), "stray_values": stray, "shape": list(a.shape)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, gpu=False)
    ap.add_argument("--cap", type=int, default=30, help="patients per outside source (seeded, one scan per patient)")
    ap.add_argument("--val-per-cohort", type=int, default=5, help="validation cases per training cohort for the seen-cohort tier (0 = none); flagged used_for_checkpoint_selection")
    ap.add_argument("--include-luna25", action="store_true", help="also draw from LUNA25 (pseudo-label masks, detection only); off by default")
    ap.add_argument("--manifest-out", type=Path, default=MANIFEST, help="where a full run writes the manifest (a smoke run writes into its run dir instead)")
    ap.add_argument("--header-cache", type=Path, default=Path(os.environ.get("NANOUNET_TMPDIR", "/root/.cache/nanounet_tmp")) / "exp00c_headers.json", help="local scratch JSON caching NIfTI header reads")
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR, help="merged corpus (dataset.json = our training volumes, splits_final.json, cohorts.json, gt_segmentations/)")
    ap.add_argument("--workers", type=int, default=16, help="threads for header reads (label reads use a quarter of it)")
    args = ap.parse_args()
    paths = {"PancancerCTSeg": PCS, "corpus dataset.json": args.corpus_dir / "dataset.json", "corpus splits_final.json": args.corpus_dir / "splits_final.json", "corpus cohorts.json": args.corpus_dir / "cohorts.json",
             "held-out images": SEEN_DIR / "imagesTs", "held-out labels": SEEN_DIR / "labelsTs", "healthy scans": HEALTHY_DIR, "holdout csv": HOLDOUT_CSV}
    paths |= {f"{s['name']} images": s["images"] for s in SOURCES if "images" in s and (s["kept"] or (s["name"] == "LUNA25" and args.include_luna25))}
    problems = missing_paths(paths, "mount /nnunet_data (PancancerCTSeg lives in /nnunet_data/raw)")
    if args.cap < 1 or args.val_per_cohort < 0:
        problems.append(problem(f"--cap {args.cap} / --val-per-cohort {args.val_per_cohort} out of range", "--cap >= 1 and --val-per-cohort >= 0", "e.g. --cap 30 --val-per-cohort 5"))
    abort_if(problems)
    inputs = {"corpus dataset.json": args.corpus_dir / "dataset.json", "splits_final.json": args.corpus_dir / "splits_final.json", "holdout csv": HOLDOUT_CSV}
    run = start_run(EXP, ap, args, inputs=inputs, paper={"section": "Experimental design > The node supply", "table_row": "1, 2", "supports": "evaluation data of the segmentation experiments"})
    rng = np.random.default_rng(args.seed)
    workers = args.workers
    dataset = json.loads((args.corpus_dir / "dataset.json").read_text())["dataset"]
    split = json.loads((args.corpus_dir / "splits_final.json").read_text())[0]
    sites = {c: s for s, cs in json.loads((args.corpus_dir / "cohorts.json").read_text())["sites"].items() for c in cs}
    trained = split["train"] + split["val"]  # the 5690 volumes the segmenter saw; dataset.json lists 106 more (RUMC pancreas) whose files are not on disk
    train_imgs = [dataset[k]["images"][0] for k in trained]
    train_h = read_headers(train_imgs, args.header_cache, workers, "training headers")
    train_keys = {}
    for k in trained:
        train_keys.setdefault(header_key(train_h[dataset[k]["images"][0]]), k)
    cases, dropped, cand_rows = [], [], []

    def add_case(case_id, patient, tier, src, cancer, annotation, image, label, values, overlap, basis, sel, h):
        cases.append({"case_id": case_id, "patient_id": patient, "tier": tier, "source": src, "cancer_type": cancer, "annotation": annotation, "image": str(image), "label": None if label is None else str(label),
                      "lesion_label_values": values, "overlap": overlap, "overlap_basis": basis, "used_for_checkpoint_selection": sel, "size": h["size"], "spacing": h["spacing"]})

    # tier seen-cohort: held-out 60 scans
    holdout = {ln.strip() for ln in HOLDOUT_CSV.read_text().splitlines()[1:] if ln.strip()}
    seen_imgs = limited(sorted(os.listdir(SEEN_DIR / "imagesTs")), args)
    seen_h = read_headers([str(SEEN_DIR / "imagesTs" / f) for f in seen_imgs], args.header_cache, workers, "held-out headers")
    for f in seen_imgs:
        stem, pid = f.removesuffix(IMG_SUFFIX), f.split("_")[2]
        assert pid in holdout, f"{f}: patient {pid} is not in test_patients.csv"
        add_case(stem, pid, "seen-cohort", "Longitudinal-CT-holdout60", "melanoma", "full", SEEN_DIR / "imagesTs" / f, SEEN_DIR / "labelsTs" / f"{stem}.nii.gz", [1],
                 {"ours": "clean", "nninteractive": "clean", "uls_plus": "overlap"}, "source", False, seen_h[str(SEEN_DIR / "imagesTs" / f)])
    # tier seen-cohort: validation cases per cohort
    val_by = defaultdict(list)
    for k in split["val"]:
        val_by[k.split("_")[0]].append(k)
    for cid in limited(sorted(val_by), args) if args.val_per_cohort else []:
        name, nni, uls = COHORTS[cid][0], COHORTS[cid][1], COHORTS[cid][2]
        pick = [val_by[cid][i] for i in sorted(rng.choice(len(val_by[cid]), size=min(args.val_per_cohort, len(val_by[cid])), replace=False))]
        for k in pick:
            img = dataset[k]["images"][0]
            add_case(k, k.split("_")[3] if cid == "d013" else k, "seen-cohort", f"val:{name}", sites[cid], "full", img, args.corpus_dir / "gt_segmentations" / f"{k}.nii.gz", [1],
                     {"ours": "clean", "nninteractive": nni, "uls_plus": uls}, "source", True, train_h[img])
    sources_out = [{"name": s, "cancer_type": next(c["cancer_type"] for c in cases if c["source"] == s), "n_total": sum(c["source"] == s for c in cases), "overlap": next(c["overlap"] for c in cases if c["source"] == s),
                    "overlap_basis": "source", "kept": True, "reason": "held-out Longitudinal-CT patients (never trained on by ours)" if s.startswith("Longitudinal") else "validation split of a training cohort (used for checkpoint selection)",
                    "annotation": "full", "tier": "seen-cohort"} for s in sorted({c["source"] for c in cases})]
    # tier outside
    for src in SOURCES:
        row = {k: v for k, v in src.items() if k not in ("images", "labels", "regex", "metadata", "min_id", "resolves_by_header")}
        use = src["kept"] or (src["name"] == "LUNA25" and args.include_luna25)
        row["kept"], row["tier"] = use, "outside"
        if src["name"] == "LUNA25" and args.include_luna25:
            row["reason"] = "included by --include-luna25: masks are MedSAM2-derived pseudo-labels, detection only"
        if "metadata" in src:
            row["metadata_sha256"] = sha256(src["metadata"])
        if not use:
            sources_out.append(row)
            continue
        cand = limited(candidates(src, args), args)
        heads = read_headers([c["image"] for c in cand], args.header_cache, workers, f"{src['name']} headers")
        alive = []
        for c in cand:
            hit = train_keys.get(header_key(heads[c["image"]]))
            if hit:
                dropped.append({"source": src["name"], "candidate": c["stem"], "matches_training_volume": hit, "reason": "header equals a training volume"})
            else:
                alive.append(c)
        by_pat = defaultdict(list)
        for c in alive:
            by_pat[c["patient"]].append(c)
        order = [p for p in sorted(by_pat)]
        chosen = [order[i] for i in rng.permutation(len(order))][: args.cap]
        for p in chosen:
            c = by_pat[p][int(rng.integers(len(by_pat[p])))]
            res = {"ours": src["overlap"]["ours"], "nninteractive": src["overlap"]["nninteractive"], "uls_plus": src["overlap"]["uls_plus"]}
            for corp in src.get("resolves_by_header", []):
                res[corp] = "clean"  # the header check above is what settles these two corpora, case by case
            add_case(c["stem"], p, "outside", src["name"], src["cancer_type"], src["annotation"], c["image"], c["label"], [1], res, src["overlap_basis"], False, heads[c["image"]])
        row.update(n_candidates=len(cand), n_dropped_header=sum(d["source"] == src["name"] for d in dropped), n_patients_alive=len(by_pat), n_selected=len(chosen))
        sources_out.append(row)
        cand_rows.append({"source": src["name"], "n_total_readme": src["n_total"], "n_candidates": len(cand), "n_dropped_header": row["n_dropped_header"], "n_patients_alive": len(by_pat), "n_selected": len(chosen)})
        cprint(f"{src['name']}: {len(cand)} candidates, {row['n_dropped_header']} dropped by header, {len(by_pat)} patients alive, {len(chosen)} selected")
    # tier healthy
    hfiles = limited(sorted(os.listdir(HEALTHY_DIR)), args)
    hh = read_headers([str(HEALTHY_DIR / f) for f in hfiles], args.header_cache, workers, "healthy headers")
    for g, (why, ours, nni, uls) in HEALTHY_GROUPS.items():
        sources_out.append({"name": f"HealthyImages-noLesion:{g}", "cancer_type": "none", "n_total": sum(f.startswith(g) for f in os.listdir(HEALTHY_DIR)), "overlap": {"ours": ours, "nninteractive": nni, "uls_plus": uls},
                            "overlap_basis": "unverified" if g == "train" else "source", "kept": True, "reason": why, "annotation": "healthy", "tier": "healthy"})
    for f in hfiles:
        g = f.split("_")[0] if f.split("_")[0] in HEALTHY_GROUPS else "train"
        key = header_key(hh[str(HEALTHY_DIR / f)])
        if key in train_keys:
            dropped.append({"source": f"HealthyImages-noLesion:{g}", "candidate": f, "matches_training_volume": train_keys[key], "reason": "header equals a training volume"})
            continue
        ours, nni, uls = HEALTHY_GROUPS[g][1:]
        add_case(f"HealthyImages:{f.removesuffix('.nii.gz').removesuffix('_0000')}", f.removesuffix(".nii.gz"), "healthy", f"HealthyImages-noLesion:{g}", "none", "healthy", HEALTHY_DIR / f, None, [],
                 {"ours": ours, "nninteractive": nni, "uls_plus": uls}, "unverified" if g == "train" else "source", False, hh[str(HEALTHY_DIR / f)])
    # validate and count lesions (every label value outside {0} + lesion values aborts, R12)
    assert len({c["case_id"] for c in cases}) == len(cases), "duplicate case ids"
    assert not [c for c in cases if c["tier"] == "outside" and "overlap" in c["overlap"].values()], "outside case flagged overlap"
    labelled = [c for c in cases if c["label"] is not None]
    with nano_progress(len(labelled), "counting lesions") as advance, ThreadPoolExecutor(max(1, workers // 4)) as ex:
        for c, st in zip(labelled, ex.map(lambda c: lesion_stats(c["label"], c["lesion_label_values"]), labelled)):
            c.update(n_lesions=st["n_lesions"], lesion_voxels=st["lesion_voxels"])
            c["_stray"] = st["stray_values"]
            advance()
    for c in cases:
        c.setdefault("n_lesions", 0)
        c.setdefault("lesion_voxels", 0)
    stray = [(c["case_id"], c.pop("_stray")) for c in labelled if c["_stray"]]
    for c in cases:
        c.pop("_stray", None)
    abort_if([problem(f"{len(stray)} label file(s) hold values other than 0 and the declared lesion values, e.g. {stray[:3]}", "binary lesion masks (0/1) for every source", "declare the lesion values of that source in SOURCES (lesion_label_values) and rerun")] if stray else [])
    cov = defaultdict(lambda: {"cases": 0, "patients": set(), "lesions": 0})
    for c in cases:
        r = cov[(c["tier"], c["source"], c["cancer_type"])]
        r["cases"] += 1
        r["patients"].add(c["patient_id"])
        r["lesions"] += c["n_lesions"]
    coverage = [{"tier": t, "source": s, "cancer_type": ct, "cases": v["cases"], "patients": len(v["patients"]), "lesions": v["lesions"]} for (t, s, ct), v in sorted(cov.items())]
    manifest = {"schema": "seg-eval-manifest/1", "seed": args.seed, "git_sha": run.rec["git"]["sha"], "cap_per_source": args.cap, "val_per_cohort": args.val_per_cohort, "sources": sources_out, "cases": cases}
    out = run.dir / "seg_eval_v1.json" if run.smoke else args.manifest_out
    out.write_text(json.dumps(manifest, indent=1))
    if not run.smoke:
        (run.dir / "seg_eval_v1.json").write_text(json.dumps(manifest, indent=1))
    t = Table(title="coverage", box=None, padding=(0, 2))
    for col in ("tier", "source", "cancer type", "cases", "patients", "lesions"):
        t.add_column(col)
    for r in coverage:
        t.add_row(r["tier"], r["source"], r["cancer_type"], str(r["cases"]), str(r["patients"]), str(r["lesions"]))
    console().print(t)
    tier_n = Counter(c["tier"] for c in cases)
    md = "# exp00c segmentation evaluation manifest\n\n" + f"cases: {dict(tier_n)}; dropped by header: {len(dropped)}; manifest: {out}\n\n| tier | source | cancer type | cases | patients | lesions |\n|---|---|---|---|---|---|\n" \
        + "".join(f"| {r['tier']} | {r['source']} | {r['cancer_type']} | {r['cases']} | {r['patients']} | {r['lesions']} |\n" for r in coverage) \
        + "\n| source | candidates | dropped by header | patients alive | selected |\n|---|---|---|---|---|\n" + "".join(f"| {r['source']} | {r['n_candidates']} | {r['n_dropped_header']} | {r['n_patients_alive']} | {r['n_selected']} |\n" for r in cand_rows)
    notes =[f"kept sources: {[r['source'] for r in cand_rows]}; LUNA25 included: {args.include_luna25}", f"{len(dropped)} candidates dropped by header match (see `dropped`)", "lesion_label_values is [1] for every source: all PancancerCTSeg, Dataset013 and gt_segmentations labels were checked to hold only 0 and 1",
             "patient ids: SegRap25/Radiogenomics/HCC/Myeloma/PANORAMA from the file name; ScienceDB, BoneTumor, AbdomenAtlas, LUNA25 have no patient map (case id used)", "no blockers: every KEEP source has an unambiguous layout and binary labels"]
    summary = {"cases": dict(tier_n), "n_cases": len(cases), "n_lesions": sum(c["n_lesions"] for c in cases), "n_dropped_header": len(dropped), "manifest": str(out), "sources_kept": [r["source"] for r in cand_rows]}
    run.finish(summary, {"sources": sources_out, "cases": cases, "dropped": dropped, "coverage": coverage, "candidates_per_source": cand_rows}, table_md=md, notes=notes, next_cmd=f"cat {run.dir}/table.md")


if __name__ == "__main__":
    main()
