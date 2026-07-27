"""Supervised patch iterable: nnUNet-aligned per-patch case draw, IO/aug overlap thread.

build_patch*/producer hand over CT crop + click COORDINATES ("keypoints"); the consumer runs the
augmentation chain on CT + points (see spatial_points.py) and only then renders heatmaps at the
final patch size -- grid_sample only ever sees CT channels (1 supervised, 2 longi). Variant
concat/split/render, click-inside bookkeeping, and collate live in patch_render.py.

Two-prompt consistency: build_patch* returns `points_variants`, one independent click draw per
prompt, all sharing ONE bbox/crop. All variants' points ride through a SINGLE augmentation pass
(concat -> augment -> split), so paired rows differ only in where the click landed -- which is what
makes the consistency term measure prompt sensitivity rather than augmentation noise. Each variant
becomes one batch row; `collate_patches` tags rows from the same patch with a shared `pair_id`.

Val-only prompt-agreement diagnostic (`emit_prompt2=True`): one extra variant is drawn per patch,
using an RNG stream fully separate from the main one, over the SAME crop/augmentation as the real
row(s). It rides the same augmentation pass but is rendered into its own `data_prompt2` tensor
instead of becoming another batch row, so val batch composition (and val_dice) is unaffected.
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
from nanounet.train.patch_render import (
    click_inside_flags,
    collate_patches,
    concat_variant_keypoints,
    render_variant,
    split_variant_keypoints,
)

__all__ = ["PatchIterable", "collate_patches", "worker_init"]

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

def _ddp_rank_world() -> tuple[int, int]:
    """(global rank, world size), 0/1 when not running under DDP."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


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
        prompts_per_patch: int = 1,
        emit_prompt2: bool = False,
    ):
        assert batch_size % prompts_per_patch == 0, (batch_size, prompts_per_patch)
        self.folder, self.keys, self.roi_cfg = folder, keys, roi_cfg
        self.patch_size, self.final_patch_size = patch_size, final_patch_size
        self.annotated_key, self.tf, self.force_zero_prompt = annotated_key, tf, force_zero_prompt
        self.num_batches, self.batch_size, self.base_seed = num_batches, batch_size, base_seed
        self.longi, self.force_null_baseline = longi, force_null_baseline
        self.prompts_per_patch = prompts_per_patch
        # Diagnostic-only extra variant (val_prompt_agreement); never affects prompts_per_patch,
        # __len__, or batch_size math -- see module docstring.
        self.emit_prompt2 = emit_prompt2

    def __len__(self) -> int:
        # item count == raw-patch count; NanoDataModule uses batch_size // prompts_per_patch.
        return self.num_batches * (self.batch_size // self.prompts_per_patch)

    def _producer(
        self,
        ds: Blosc2Folder,
        q: queue.Queue,
        n_here: int,
        rng: np.random.Generator,
        meta: CaseMetaCache,
        stop: threading.Event,
        extra_rng: np.random.Generator | None,
    ) -> None:
        try:
            for _ in range(n_here):
                if stop.is_set(): break
                cid = self.keys[int(rng.integers(0, len(self.keys)))]
                prop = meta.get(cid) or meta.put(cid, load_case_properties(ds.source_folder, cid))
                with ds.open_case(cid, need_seg=True) as (data, seg, _, _):
                    common = (data, seg, prop, self.roi_cfg, self.patch_size, self.final_patch_size)
                    n = self.prompts_per_patch
                    if self.longi:
                        raw = build_patch_longi(*common, self.force_zero_prompt, self.force_null_baseline, rng, n, extra_rng=extra_rng)
                    else:
                        raw = build_patch(*common, self.annotated_key, self.force_zero_prompt, rng, n, extra_rng=extra_rng)
                q.put(raw)
        except Exception as e:
            q.put(e)
        finally:
            q.put(None)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        nw = 1 if wi is None else wi.num_workers
        wid = 0 if wi is None else wi.id
        # Seed by (rank, worker), not worker alone. Under DDP every rank gets the same base_seed and
        # the same worker ids, so seeding by worker alone makes every rank draw the IDENTICAL patch
        # sequence: it does not crash, it silently trains each step on world_size copies of the same
        # data, and no loss curve reveals it. Cases are drawn i.i.d. with replacement rather than
        # partitioned, so a distinct stream per (rank, worker) is the whole fix.
        # n_here stays PER RANK: every rank yields the full item count, so an epoch keeps its
        # iters-per-epoch step count and each step sees world_size * batch_size rows.
        rank, world = _ddp_rank_world()
        shard = rank * nw + wid
        total = len(self)  # per-rank item count == raw-patch count
        n_here = total // nw + (1 if wid < (total % nw) else 0)
        rng = np.random.default_rng(self.base_seed + shard * 10007)
        # Separate RNG stream for the diagnostic 2nd prompt (never touches `rng` above), so the
        # primary patch/prompt sequence -- and therefore val_dice -- is unaffected by emit_prompt2.
        extra_rng = np.random.default_rng(self.base_seed + shard * 10007 + 777_777) if self.emit_prompt2 else None
        ds = Blosc2Folder(self.folder, identifiers=self.keys)
        meta = CaseMetaCache()
        stats = {"patches": 0}
        q: queue.Queue = queue.Queue(maxsize=_QUEUE_DEPTH)
        stop = threading.Event()
        prod = threading.Thread(target=self._producer, args=(ds, q, n_here, rng, meta, stop, extra_rng), daemon=True)
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
                kp = concat_variant_keypoints(variants, self.longi)
                with torch.no_grad():
                    o = self.tf(**{"image": im, "segmentation": se, "keypoints": kp})
                    entries = split_variant_keypoints(o["keypoints"], variants, self.longi)
                    # entries beyond prompts_per_patch (if any) are the emit_prompt2 diagnostic
                    # draw -- rendered separately below, never exploded into a batch row.
                    real_entries = entries[: self.prompts_per_patch]
                    data_variants = [render_variant(o, e, raw, self.longi, self.final_patch_size, self.roi_cfg.prompt) for e in real_entries]
                    seg0 = o["segmentation"][0] if isinstance(o["segmentation"], list) else o["segmentation"]
                    click_inside = click_inside_flags(real_entries, seg0)
                item = {"data_variants": data_variants, "target": o["segmentation"], "click_inside": click_inside}
                if self.emit_prompt2:
                    item["data_prompt2"] = render_variant(o, entries[-1], raw, self.longi, self.final_patch_size, self.roi_cfg.prompt)
                yield item
                stats["patches"] += 1
        finally:
            stop.set()
            prod.join(timeout=30.0)
