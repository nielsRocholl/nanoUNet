"""Fixed validation manifest: schema, load, and a deterministic map-style patch dataset.

The training val loader re-samples patches every epoch, so per-scenario curves drown in resampling
noise. This module reads a manifest built offline by nanounet_build_valset instead: every patch is
pinned by (case, bbox, click coordinates), both prompt draws are stored, and nothing is randomised
at validation time -- two runs on one manifest give bit-identical metrics.

Scenario targets: all_clicked and subset_clicked score against seg; none_clicked and
lesion_free_decoy score predicted-foreground fraction (correct output is empty, so Dice is
undefined). subset_clicked additionally carries a precomputed clicked-subset target, packed to
bits in a sidecar .npz -- that is what keeps cc3d off the validation path entirely.

Per-cohort metrics are defined over all_clicked rows only: single-lesion cohorts (e.g. d014/d016/
d020) get zero subset_clicked patches by construction (see nanounet_build_valset), so mixing
scenarios into one per-cohort number would compare scenario difficulty rather than model quality.

Every entry is baked in at build time under a specific RoiPromptConfig; load_manifest stamps that
config into the header and refuses a stale mismatch -- see config_stamp() for which fields and why."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json
from torch.utils.data import DataLoader, Dataset

from nanounet.config import RoiPromptConfig
from nanounet.data.blosc2_dataset import Blosc2Folder
from nanounet.data.patch_bbox import crop_patch
from nanounet.dataloader_prefs import DataloaderBucket, build_iter_dataloader
from nanounet.train.patch_iterable import collate_patches, worker_init
from nanounet.train.patch_render import concat_variant_keypoints, render_variant, split_variant_keypoints

SCHEMA_VERSION = 1
SCENARIOS = ("all_clicked", "subset_clicked", "none_clicked", "lesion_free_decoy")
SIZE_BUCKETS = ("small", "large")
SMALL_LESION_MAX_VOX = 500  # ~10mm diameter at the plans spacing; see docs/steps/valset.md

@dataclass(frozen=True)
class ValManifest:
    path: str
    header: dict
    entries: list[dict]
    packed: np.ndarray | None  # (n, nbytes) uint8, or None when no subset entries
    patch_size: tuple[int, int, int]

def _sidecar_path(manifest_path: str) -> str:
    if not manifest_path.endswith(".json"):
        raise ValueError(f"manifest path must end in .json, got {manifest_path!r}")
    return manifest_path[: -len(".json")] + ".targets.npz"

def config_stamp(cfg: RoiPromptConfig) -> dict:
    """Fields nanounet_build_valset reads while choosing WHICH patch/clicks to bake into a row
    (draw_lesion_clicks in valset_build.py). Only sampling.propagated qualifies -- it drives the
    per-lesion displacement behind the stored click coordinates. Everything else in RoiPromptConfig
    either never reaches the build (fg_patch_prob, click_modes, false_pos_probability,
    instance_targets are fixed by the build's own scenario logic) or is applied fresh at load time
    regardless of the manifest (prompt.* rendering uses the LIVE roi_cfg in ValPatchDataset), so a
    mismatch there can't make the manifest stale."""
    return {"propagated": asdict(cfg.sampling.propagated)}

def load_manifest(path: str, cfg: RoiPromptConfig) -> ValManifest:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No validation manifest at {path}.\n"
            f"Expected the output of nanounet_build_valset.\n"
            f"Fix: nanounet_build_valset -d 999 --plans <plans> --config configs/default.json "
            f"--out {path}   (see docs/steps/valset.md)"
        )
    header = load_json(path)
    if header.get("schema") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema {header.get('schema')}, this build expects {SCHEMA_VERSION}.\n"
            f"Fix: rebuild it with nanounet_build_valset"
        )
    stamp = header.get("config_stamp")
    if stamp is None:
        raise ValueError(
            f"{path} has no config_stamp (built by an older nanounet_build_valset).\n"
            f"Without it a manifest built under a different sampling config loads silently, and "
            f"the val curve becomes meaningless.\n"
            f"Fix: nanounet_build_valset -d <id> --plans <plans> --config <cfg> --out {path}"
        )
    live = config_stamp(cfg)
    if stamp != live:
        keys = sorted(set(stamp) | set(live))
        diff = "\n".join(f"  {k}: manifest={stamp.get(k)!r} vs live={live.get(k)!r}" for k in keys if stamp.get(k) != live.get(k))
        raise ValueError(
            f"{path} was built under a different sampling config than the one now in use:\n{diff}\n"
            f"Fix: nanounet_build_valset -d <id> --plans <plans> --config <cfg> --out {path}"
        )
    entries = header["entries"]
    patch_size = tuple(int(x) for x in header["patch_size"])
    npz_path = _sidecar_path(path)
    needs_packed = any(e["subset_target_index"] >= 0 for e in entries)
    packed = None
    if needs_packed:
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(
                f"{path} has subset_clicked entries but its sidecar {npz_path} is missing.\n"
                f"Fix: rebuild with nanounet_build_valset (never hand-edit a manifest)"
            )
        packed = np.load(npz_path)["packed"]
    return ValManifest(path=path, header=header, entries=entries, packed=packed, patch_size=patch_size)

class ValPatchDataset(Dataset):
    """Map-style val dataset over a fixed ValManifest. __getitem__ re-crops from the case, re-runs
    val_tf (no spatial transform -- see augment.py), and renders both stored prompt draws. No RNG
    anywhere here: every random choice already happened in nanounet_build_valset."""

    def __init__(
        self,
        manifest: ValManifest,
        case_folder: str,
        roi_cfg: RoiPromptConfig,
        val_tf,
        final_patch_size,
        longi: bool,
    ):
        self.manifest = manifest
        self.pr = roi_cfg.prompt
        self.tf = val_tf
        self.final_ps = final_patch_size
        self.patch_size = manifest.patch_size
        self.longi = longi
        cases = sorted({e["case"] for e in manifest.entries})
        self.ds = Blosc2Folder(case_folder, identifiers=cases)
        # Shared with val_metrics.py, which reads the same cohort_weights keys off the manifest
        # header, so the integer<->name mapping is stable across the two modules.
        self.cohort_index = {c: i for i, c in enumerate(sorted(manifest.header["cohort_weights"]))}

    def __len__(self) -> int:
        return len(self.manifest.entries)

    def __getitem__(self, i: int) -> dict:
        e = self.manifest.entries[i]
        with self.ds.open_case(e["case"], need_seg=True) as (data, seg, _, _):
            data_crop, seg_crop, _shape, _pslc = crop_patch(data, seg, e["bbox"])
        im = torch.from_numpy(data_crop.astype(np.float32)).float()
        se = torch.from_numpy(seg_crop.astype(np.int16)).short()

        # Both draws ride ONE augmentation pass, exactly as PatchIterable does, so the pair differs
        # only in click placement.
        variants = [
            {"points_pos": np.asarray(e["clicks_zyx"], np.float32).reshape(-1, 3),
             "points_neg": np.zeros((0, 3), np.float32), "n_false_pos": e["n_false_pos"]},
            {"points_pos": np.asarray(e["clicks2_zyx"], np.float32).reshape(-1, 3),
             "points_neg": np.zeros((0, 3), np.float32), "n_false_pos": e["n_false_pos"]},
        ]
        kp = concat_variant_keypoints(variants, self.longi)
        with torch.no_grad():
            o = self.tf(**{"image": im, "segmentation": se, "keypoints": kp})
            split = split_variant_keypoints(o["keypoints"], variants, self.longi)
            v1 = render_variant(o, split[0], {"null_baseline": False}, self.longi, self.final_ps, self.pr)
            v2 = render_variant(o, split[1], {"null_baseline": False}, self.longi, self.final_ps, self.pr)
        item = {
            "data_variants": [v1], "data_prompt2": v2, "target": o["segmentation"],
            "click_inside": [e["click_inside"]], "scenario": SCENARIOS.index(e["scenario"]),
            "cohort": self.cohort_index[e["cohort"]], "size_bucket": SIZE_BUCKETS.index(e["size_bucket"]),
            # Displacement can push a lesion click out of the patch in one draw but not the other
            # (67/1500 on the real manifest); those rows measure "a lesion left the prompt", not
            # pure placement jitter -- val_metrics reports both variants.
            "draws_matched": int(len(e["clicks_zyx"]) == len(e["clicks2_zyx"])),
        }
        if e["subset_target_index"] >= 0:
            bits = np.unpackbits(self.manifest.packed[e["subset_target_index"]])
            m = bits[: int(np.prod(self.patch_size))].reshape(self.patch_size)
            item["target_subset"] = torch.from_numpy(m.astype(np.int16))[None]
        else:
            item["target_subset"] = torch.zeros((1, *self.patch_size), dtype=torch.int16)
        item["has_subset"] = int(e["subset_target_index"] >= 0)
        return item

def build_val_dataloader(
    manifest: ValManifest,
    case_folder: str,
    roi_cfg: RoiPromptConfig,
    val_tf,
    final_ps,
    longi: bool,
    batch_size: int,
    bucket: DataloaderBucket,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    """Deterministic val loader over a fixed manifest. shuffle stays False and no sampler is
    passed: Lightning injects a DistributedSampler under DDP, and order does not affect any
    metric here because every bucket is pooled before reduction."""
    ds = ValPatchDataset(manifest, case_folder, roi_cfg, val_tf, final_ps, longi)
    nw = bucket.nw_val
    winit = worker_init if nw else None
    return build_iter_dataloader(
        ds, batch_size=batch_size, bucket=bucket, nw=nw, prefetch=bucket.prefetch_val,
        collate_fn=collate_patches, pin_memory=pin_memory, worker_init_fn=winit,
        persistent_workers=persistent_workers,
    )
