"""Test-time augmentation: average logits over jittered graph copies."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch

from tracking.data.features import FeatConfig
from tracking.data.augment import jitter_both
from tracking.matcher import MatcherOutput


def matcher_tta_forward(
    matcher: torch.nn.Module,
    batch: Batch,
    n: int,
    k_intra: int,
    fu_jitter: float,
    desc_jitter_frac: float,
) -> MatcherOutput:
    if n <= 0:
        return matcher(batch)
    dev = next(matcher.parameters()).device
    batch = batch.to(dev)
    outs: list[MatcherOutput] = []
    for _ in range(n):
        gl = []
        for g in batch.to_data_list():
            gc = g.clone()
            jitter_both(
                gc,
                k_intra=k_intra,
                sigma_fu_scale=fu_jitter,
                rng=np.random.default_rng(),
                desc_jitter_frac=desc_jitter_frac,
                feat=FeatConfig(mode=str(getattr(gc, "feat_mode", "l0"))),
            )
            gl.append(gc)
        b2 = Batch.from_data_list(gl).to(dev)
        outs.append(matcher(b2))
    pair = torch.stack([o.pair for o in outs]).mean(0)
    dust_bl = torch.stack([o.dust_bl for o in outs]).mean(0)
    dust_fu = torch.stack([o.dust_fu for o in outs]).mean(0)
    zb, zf = outs[-1].z_bl, outs[-1].z_fu
    return MatcherOutput(pair, dust_bl, dust_fu, zb, zf)
