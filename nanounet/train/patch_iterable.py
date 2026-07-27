"""Supervised patch iterable: nnUNet-aligned per-patch case draw, IO/aug overlap thread.

build_patch*/producer hand over CT crop + click COORDINATES ("keypoints"); the consumer runs the
augmentation chain on CT + points (SpatialPointsTransform/MirrorPointsTransform move the points
alongside the image, see nanounet/data/spatial_points.py) and only then renders heatmaps at the
final patch size, assembling the image tensor the model actually sees. This keeps grid_sample in
the chain to CT channels only (1 supervised, 2 longi) instead of also warping heatmap channels.
"""

from __future__ import annotations

import queue
import threading
from typing import List

import numpy as np
import torch
from torch.utils.data import IterableDataset

from nanounet.config import RoiPromptConfig
from nanounet.data.blosc2_dataset import Blosc2Folder, load_case_properties
from nanounet.data.sampling import build_patch
from nanounet.data.sampling_longi import build_patch_longi
from nanounet.dataloader_prefs import pin_worker_threads
from nanounet.prompt.encoding import encode_points_to_heatmap_pair

_QUEUE_DEPTH = 2
_META_CAP = 512


class CaseMetaCache:
    def __init__(self, cap: int = _META_CAP):
        self._cap = cap
        self._d: dict[str, dict] = {}

    def get(self, cid: str) -> dict | None:
        return self._d.get(cid)

    def put(self, cid: str, prop: dict) -> dict:
        if cid not in self._d and len(self._d) >= self._cap:
            self._d.pop(next(iter(self._d)))
        self._d[cid] = prop
        return prop


def worker_init(worker_id: int) -> None:
    from nanounet.runtime import set_safe_tmpdir

    set_safe_tmpdir()
    pin_worker_threads()


def _point_list(pts: torch.Tensor) -> list:
    if pts.numel() == 0:
        return []
    return [tuple(v) for v in torch.round(pts).long().tolist()]


def _render(o: dict, raw: dict, longi: bool, final_patch_size, pr) -> torch.Tensor:
    """Split the augmented `keypoints` back into its per-stream groups (same concat order used
    in __iter__ below) and render heatmaps at the final patch size."""
    shape = tuple(int(s) for s in final_patch_size)
    kp = o["keypoints"]
    n_pp, n_pn = raw["points_pos"].shape[0], raw["points_neg"].shape[0]
    fu_pp = _point_list(kp[:n_pp])
    fu_pn = _point_list(kp[n_pp : n_pp + n_pn])
    fu_hm = encode_points_to_heatmap_pair(fu_pp, fu_pn, shape, pr.point_radius_vox, pr.encoding, None, pr.prompt_intensity_scale)
    if not longi:
        return torch.cat([o["image"][0:1], fu_hm], dim=0)

    fu_stream = torch.cat([o["image"][0:1], fu_hm], dim=0)
    if raw["null_baseline"]:
        bl_stream = fu_stream  # duplicate the RENDERED FU stream -> identity DWB (matches old behaviour)
    else:
        n_bp = raw["bl_points_pos"].shape[0]
        bl_pp = _point_list(kp[n_pp + n_pn : n_pp + n_pn + n_bp])
        bl_hm = encode_points_to_heatmap_pair(bl_pp, [], shape, pr.point_radius_vox, pr.encoding, None, pr.prompt_intensity_scale)
        bl_stream = torch.cat([o["image"][1:2], bl_hm], dim=0)
    return torch.cat([fu_stream, bl_stream], dim=0)


class PatchIterable(IterableDataset):
    def __init__(
        self,
        folder: str,
        keys: List[str],
        roi_cfg: RoiPromptConfig,
        patch_size: np.ndarray,
        final_patch_size: np.ndarray,
        annotated_key,
        tf,
        force_zero_prompt: bool,
        num_batches: int,
        batch_size: int,
        base_seed: int,
        longi: bool = False,
        force_null_baseline: bool = False,
    ):
        self.folder = folder
        self.keys = keys
        self.roi_cfg = roi_cfg
        self.patch_size = patch_size
        self.final_patch_size = final_patch_size
        self.annotated_key = annotated_key
        self.tf = tf
        self.force_zero_prompt = force_zero_prompt
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.base_seed = base_seed
        self.longi = longi
        self.force_null_baseline = force_null_baseline

    def __len__(self) -> int:
        return self.num_batches * self.batch_size

    def _producer(
        self,
        ds: Blosc2Folder,
        q: queue.Queue,
        n_here: int,
        rng: np.random.Generator,
        meta: CaseMetaCache,
        stats: dict,
        stop: threading.Event,
    ) -> None:
        try:
            for _ in range(n_here):
                if stop.is_set():
                    break
                cid = self.keys[int(rng.integers(0, len(self.keys)))]
                prop = meta.get(cid)
                if prop is None:
                    prop = meta.put(cid, load_case_properties(ds.source_folder, cid))
                with ds.open_case(cid, need_seg=True) as (data, seg, _, _):
                    if self.longi:
                        raw = build_patch_longi(
                            data, seg, prop, self.roi_cfg, self.patch_size, self.final_patch_size,
                            self.force_zero_prompt, self.force_null_baseline, rng,
                        )
                    else:
                        raw = build_patch(
                            data, seg, prop, self.roi_cfg, self.patch_size, self.final_patch_size,
                            self.annotated_key, self.force_zero_prompt, rng,
                        )
                q.put(raw)
        except Exception as e:
            q.put(e)
        finally:
            q.put(None)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        nw = 1 if wi is None else wi.num_workers
        wid = 0 if wi is None else wi.id
        total = self.num_batches * self.batch_size
        n_here = total // nw + (1 if wid < (total % nw) else 0)
        rng = np.random.default_rng(self.base_seed + wid * 10007)
        ds = Blosc2Folder(self.folder, identifiers=self.keys)
        meta = CaseMetaCache()
        stats = {"patches": 0}
        q: queue.Queue = queue.Queue(maxsize=_QUEUE_DEPTH)
        stop = threading.Event()
        prod = threading.Thread(
            target=self._producer,
            args=(ds, q, n_here, rng, meta, stats, stop),
            daemon=True,
        )
        prod.start()
        try:
            while stats["patches"] < n_here:
                raw = q.get()
                if raw is None:
                    break
                if isinstance(raw, Exception):
                    raise raw
                im = torch.from_numpy(raw["image"]).float()
                se = torch.from_numpy(raw["segmentation"]).short()
                kp_parts = [raw["points_pos"], raw["points_neg"]]
                if self.longi:
                    kp_parts.append(raw["bl_points_pos"])
                kp = torch.from_numpy(np.concatenate(kp_parts, axis=0)).float()
                with torch.no_grad():
                    o = self.tf(**{"image": im, "segmentation": se, "keypoints": kp})
                    data = _render(o, raw, self.longi, self.final_patch_size, self.roi_cfg.prompt)
                yield {"data": data, "target": o["segmentation"]}
                stats["patches"] += 1
        finally:
            stop.set()
            prod.join(timeout=30.0)


def collate_patches(batch: list) -> dict:
    data = torch.stack([b["data"] for b in batch])
    t0 = batch[0]["target"]
    if isinstance(t0, list):
        target = [torch.stack([b["target"][i] for b in batch], dim=0) for i in range(len(t0))]
    else:
        target = torch.stack([b["target"] for b in batch])
    return {"data": data, "target": target}
