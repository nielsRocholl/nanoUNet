"""Supervised patch iterable: nnUNet-aligned per-patch case draw, IO/aug overlap thread.

build_patch*/producer hand over CT crop + click COORDINATES ("keypoints"); the consumer runs the
augmentation chain on CT + points (see spatial_points.py) and only then renders heatmaps at the
final patch size -- grid_sample only ever sees CT channels (1 supervised, 2 longi).

Two-prompt consistency (prompts_per_patch > 1): the producer draws `prompts_per_patch` independent
click sets per patch under raw["points_variants"] (list of dicts shaped like the old single-draw
fields). One raw patch = one case draw + one augmentation pass; only rendered heatmaps differ
across variants, isolating prompt- from augmentation-variance. Each raw patch yields ONE item
holding all its variants; collate_patches flattens them into rows with a shared `pair_id` so the
consistency loss can regroup them after one forward pass.
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

def _concat_variant_keypoints(variants: list, longi: bool) -> torch.Tensor:
    """Concat every variant's clicks into one (N,3) tensor, order (pos, neg[, bl_pos]) per
    variant, so a single augmentation pass moves all of them together."""
    parts = []
    for v in variants:
        parts += [v["points_pos"], v["points_neg"]] + ([v["bl_points_pos"]] if longi else [])
    if not parts:
        return torch.zeros((0, 3), dtype=torch.float32)
    return torch.from_numpy(np.concatenate(parts, axis=0)).float()

def _split_variant_keypoints(kp: torch.Tensor, variants: list, longi: bool) -> list:
    """Inverse of _concat_variant_keypoints: slice augmented `keypoints` back per variant."""
    out, off = [], 0
    for v in variants:
        n_pp, n_pn = v["points_pos"].shape[0], v["points_neg"].shape[0]
        pp, off = kp[off : off + n_pp], off + n_pp
        pn, off = kp[off : off + n_pn], off + n_pn
        entry = {"pp": pp, "pn": pn}
        if longi:
            n_bp = v["bl_points_pos"].shape[0]
            entry["bp"], off = kp[off : off + n_bp], off + n_bp
        out.append(entry)
    return out

def _render_variant(o: dict, entry: dict, raw: dict, longi: bool, final_patch_size, pr) -> torch.Tensor:
    shape = tuple(int(s) for s in final_patch_size)
    fu_hm = encode_points_to_heatmap_pair(
        _point_list(entry["pp"]), _point_list(entry["pn"]), shape, pr.point_radius_vox, pr.encoding, None, pr.prompt_intensity_scale
    )
    if not longi:
        return torch.cat([o["image"][0:1], fu_hm], dim=0)
    fu_stream = torch.cat([o["image"][0:1], fu_hm], dim=0)
    if raw["null_baseline"]:
        bl_stream = fu_stream  # duplicate the RENDERED FU stream -> identity DWB (matches old behaviour)
    else:
        bl_hm = encode_points_to_heatmap_pair(_point_list(entry["bp"]), [], shape, pr.point_radius_vox, pr.encoding, None, pr.prompt_intensity_scale)
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
        prompts_per_patch: int = 1,
    ):
        assert batch_size % prompts_per_patch == 0, (batch_size, prompts_per_patch)
        self.folder, self.keys, self.roi_cfg = folder, keys, roi_cfg
        self.patch_size, self.final_patch_size = patch_size, final_patch_size
        self.annotated_key, self.tf, self.force_zero_prompt = annotated_key, tf, force_zero_prompt
        self.num_batches, self.batch_size, self.base_seed = num_batches, batch_size, base_seed
        self.longi, self.force_null_baseline = longi, force_null_baseline
        self.prompts_per_patch = prompts_per_patch

    def __len__(self) -> int:
        # Item count (1 item/raw patch, `prompts_per_patch` rows each) -- NanoDataModule sets the
        # DataLoader batch_size to batch_size // prompts_per_patch so a batch has `batch_size` rows.
        return self.num_batches * (self.batch_size // self.prompts_per_patch)

    def _producer(self, ds: Blosc2Folder, q: queue.Queue, n_here: int, rng: np.random.Generator, meta: CaseMetaCache, stop: threading.Event) -> None:
        try:
            for _ in range(n_here):
                if stop.is_set():
                    break
                cid = self.keys[int(rng.integers(0, len(self.keys)))]
                prop = meta.get(cid) or meta.put(cid, load_case_properties(ds.source_folder, cid))
                with ds.open_case(cid, need_seg=True) as (data, seg, _, _):
                    common = (data, seg, prop, self.roi_cfg, self.patch_size, self.final_patch_size)
                    if self.longi:
                        raw = build_patch_longi(*common, self.force_zero_prompt, self.force_null_baseline, rng, prompts_per_patch=self.prompts_per_patch)
                    else:
                        raw = build_patch(*common, self.annotated_key, self.force_zero_prompt, rng, prompts_per_patch=self.prompts_per_patch)
                q.put(raw)
        except Exception as e:
            q.put(e)
        finally:
            q.put(None)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        nw = 1 if wi is None else wi.num_workers
        wid = 0 if wi is None else wi.id
        total = len(self)  # item count == raw-patch count
        n_here = total // nw + (1 if wid < (total % nw) else 0)
        rng = np.random.default_rng(self.base_seed + wid * 10007)
        ds = Blosc2Folder(self.folder, identifiers=self.keys)
        meta = CaseMetaCache()
        stats = {"patches": 0}
        q: queue.Queue = queue.Queue(maxsize=_QUEUE_DEPTH)
        stop = threading.Event()
        prod = threading.Thread(target=self._producer, args=(ds, q, n_here, rng, meta, stop), daemon=True)
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
                variants = raw["points_variants"]
                kp = _concat_variant_keypoints(variants, self.longi)
                with torch.no_grad():
                    o = self.tf(**{"image": im, "segmentation": se, "keypoints": kp})
                    entries = _split_variant_keypoints(o["keypoints"], variants, self.longi)
                    data_variants = [_render_variant(o, e, raw, self.longi, self.final_patch_size, self.roi_cfg.prompt) for e in entries]
                yield {"data_variants": data_variants, "target": o["segmentation"]}
                stats["patches"] += 1
        finally:
            stop.set()
            prod.join(timeout=30.0)

def collate_patches(batch: list) -> dict:
    """Flatten each item's `prompts_per_patch` variants into rows; `pair_id` groups rows from the
    same raw patch (same case draw + augmentation, different rendered clicks)."""
    t0 = batch[0]["target"]
    is_list = isinstance(t0, list)
    rows_data, rows_target, pair_ids = [], [], []
    for pid, item in enumerate(batch):
        for v in item["data_variants"]:
            rows_data.append(v)
            rows_target.append(item["target"])
            pair_ids.append(pid)
    data = torch.stack(rows_data)
    if is_list:
        target = [torch.stack([t[i] for t in rows_target], dim=0) for i in range(len(t0))]
    else:
        target = torch.stack(rows_target)
    return {"data": data, "target": target, "pair_id": torch.tensor(pair_ids, dtype=torch.long)}
