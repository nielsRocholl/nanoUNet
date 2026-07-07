"""Supervised patch iterable: nnUNet-aligned per-patch case draw, IO/aug overlap thread."""

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
                            data,
                            seg,
                            prop,
                            self.roi_cfg,
                            self.patch_size,
                            self.final_patch_size,
                            self.force_zero_prompt,
                            self.force_null_baseline,
                            rng,
                        )
                    else:
                        raw = build_patch(
                            data,
                            seg,
                            prop,
                            self.roi_cfg,
                            self.patch_size,
                            self.final_patch_size,
                            self.annotated_key,
                            self.force_zero_prompt,
                            rng,
                        )
                q.put((raw["image"], raw["segmentation"]))
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
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                im_np, se_np = item
                im = torch.from_numpy(im_np).float()
                se = torch.from_numpy(se_np).short()
                with torch.no_grad():
                    o = self.tf(**{"image": im, "segmentation": se})
                yield {"data": o["image"], "target": o["segmentation"]}
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
