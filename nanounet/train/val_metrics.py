"""Validation metric logging: aggregate, per scenario, per source dataset, per tag.

Lives outside lightning_module.py, which is at the 200-LOC ceiling. Everything here is tensor
arithmetic over buffers already in host memory -- no forward passes, no device syncs beyond the
ones the caller already paid for.

Two reporting layers, deliberately asymmetric:
  * scenario  -- the full metric set, because each scenario tests a different behaviour;
  * cohort    -- val_dice and val_prompt_agreement only, because 17 datasets x 9 metrics is
                 unreadable and those two answer "how accurate" and "how stable".
Headline aggregates are re-weighted to the true cohort proportions, undoing the per-cohort patch
floor the manifest builder applies to keep small cohorts plottable.

`draws_matched` (see valset.py) splits val_prompt_agreement into the historical, all-rows number
and a `_matched` variant restricted to rows where displacement did not drop a lesion from either
prompt draw -- isolating pure click-placement sensitivity from "a lesion left the prompt"."""

from __future__ import annotations

import numpy as np
import torch

from nanounet.data.valset import SCENARIOS, SIZE_BUCKETS
from nanounet.model.dice_helpers import agreement_mean, click_split_means, pooled_dice_from_rows, pooled_fg_dice


def _dice_sel(tp, fp, fn, sel):
    return pooled_dice_from_rows(tp[sel], fp[sel], fn[sel])


def _agree_sel(agree, sel):
    a = agree[sel]
    a = a[~torch.isnan(a)]
    return float(a.mean()) if a.numel() else float("nan")


def _mean_sel(x, sel):
    a = x[sel]
    return float(a.mean()) if a.numel() else float("nan")


def _weighted_mean(vals: dict, weights: dict) -> float:
    items = [(v, weights[k]) for k, v in vals.items() if not np.isnan(v)]
    s = sum(w for _, w in items)
    return float(sum(v * w for v, w in items) / s) if s > 0 else float("nan")


def log_val_metrics(lm) -> None:
    """Called from NanoUNetLM.on_validation_epoch_end. Reads lm._val_buf, lm._val_buf_ablated,
    lm._agreement_buf, lm._meta_buf and calls lm.log(...)."""
    d = dict(sync_dist=True)  # each rank validates its own shard; else metrics are rank 0 only
    da, fb = (torch.cat([v[k] for v in lm._val_buf]) for k in ("dice_a", "fp_b"))
    val_dice = pooled_fg_dice(lm._val_buf)
    lm.log("val_dice", val_dice, prog_bar=True, **d)
    lm.log("val_dice_macro", float(da.mean()) if da.numel() else float("nan"), prog_bar=True, **d)
    lm.log("val_fp", float(fb.mean()) if fb.numel() else 0.0, prog_bar=False, **d)
    for k, v in (("val_n_a", da.numel()), ("val_n_b", fb.numel())):
        lm.log(k, float(v), reduce_fx="sum", sync_dist=True)
    lm.log("val_loss", float(np.mean([v["loss"] for v in lm._val_buf])), prog_bar=False, **d)
    if lm._val_buf_ablated:
        val_dice_ablated = pooled_fg_dice(lm._val_buf_ablated)
        lm.log("val_dice_prompt_ablated", val_dice_ablated, **d)
        lm.log("val_prompt_gap", val_dice - val_dice_ablated, **d)  # METRIC 3: collapse guard
    din, dout = click_split_means(lm._val_buf)
    lm.log("val_dice_click_inside", din, **d)
    lm.log("val_dice_click_outside", dout, **d)
    lm.log("val_prompt_agreement", agreement_mean(lm._agreement_buf), **d)  # METRIC 1 (headline)

    if not lm._meta_buf:  # training run without --val-manifest: legacy path stops here
        return

    scenario = torch.cat([m["scenario"] for m in lm._meta_buf])
    cohort = torch.cat([m["cohort"] for m in lm._meta_buf])
    size_bucket = torch.cat([m["size_bucket"] for m in lm._meta_buf])
    has_subset = torch.cat([m["has_subset"] for m in lm._meta_buf])
    draws_matched = torch.cat([m["draws_matched"] for m in lm._meta_buf])
    click_in = torch.cat([m["click_inside"] for m in lm._meta_buf])
    n = scenario.numel()
    agree = torch.cat(lm._agreement_buf) if lm._agreement_buf else torch.full((n,), float("nan"))
    assert agree.numel() == n, (agree.numel(), n)

    tp, fp, fn = (torch.cat([v[k] for v in lm._val_buf]) for k in ("tp_row", "fp_row", "fn_row"))
    pred_fg = torch.cat([v["pred_fg_row"] for v in lm._val_buf])
    tp_a, fp_a, fn_a = (torch.cat([v[k] for v in lm._val_buf_ablated]) for k in ("tp_row", "fp_row", "fn_row"))
    assert tp.shape[0] == n, (tp.shape[0], n)
    has_fg = (tp + fn).sum(1) > 0
    den = 2 * tp.sum(1) + fp.sum(1) + fn.sum(1)
    dice_row = torch.where(den > 0, 2 * tp.sum(1) / den, torch.zeros_like(den))

    tp_s, fp_s, fn_s = (torch.zeros_like(tp) for _ in range(3))
    off = 0
    for m in lm._meta_buf:
        b = len(m["scenario"])
        if "subset_row" in m:
            tp_s[off : off + b], fp_s[off : off + b], fn_s[off : off + b] = m["subset_row"]
        off += b

    # (c) per scenario
    for si, s in enumerate(SCENARIOS):
        sel = scenario == si
        lm.log(f"val/{s}/n", float(sel.sum()), reduce_fx="sum", sync_dist=True)
        lm.log(f"val/{s}/val_pred_fg", _mean_sel(pred_fg, sel), **d)  # D10: lower is better
        lm.log(f"val/{s}/val_prompt_agreement", _agree_sel(agree, sel), **d)
        matched = sel & (draws_matched == 1)
        lm.log(f"val/{s}/val_prompt_agreement_matched", _agree_sel(agree, matched), **d)
        if s in ("all_clicked", "subset_clicked"):
            dice_s = _dice_sel(tp, fp, fn, sel)
            lm.log(f"val/{s}/val_dice", dice_s, **d)
            lm.log(f"val/{s}/val_dice_macro", _mean_sel(dice_row, sel & has_fg), **d)
            dice_ab = _dice_sel(tp_a, fp_a, fn_a, sel)
            lm.log(f"val/{s}/val_dice_prompt_ablated", dice_ab, **d)
            lm.log(f"val/{s}/val_prompt_gap", dice_s - dice_ab, **d)
    lm.log("val_prompt_agreement_matched", _agree_sel(agree, draws_matched == 1), **d)

    # (d) subset diagnostic -- headline number of this step
    sub_sel = has_subset == 1
    dice_clicked = _dice_sel(tp_s, fp_s, fn_s, sub_sel)
    dice_all = _dice_sel(tp, fp, fn, sub_sel)
    lm.log("val/subset_clicked/val_dice_vs_clicked_subset", dice_clicked, **d)
    lm.log("val/subset_clicked/val_dice_vs_all_lesions", dice_all, **d)
    lm.log("val/subset_clicked/val_selectivity_margin", dice_clicked - dice_all, **d)

    # (e) per cohort (all_clicked rows only, D15) + (g) weighted headline
    manifest = lm.trainer.datamodule.val_manifest
    cohort_weights = manifest.header["cohort_weights"]
    cohort_names = sorted(cohort_weights)
    ac_sel = scenario == SCENARIOS.index("all_clicked")
    dice_by_cohort, agree_by_cohort = {}, {}
    for ci, name in enumerate(cohort_names):
        sel = ac_sel & (cohort == ci)
        lm.log(f"val/cohort/{name}/n", float(sel.sum()), reduce_fx="sum", sync_dist=True)
        dice_by_cohort[name] = _dice_sel(tp, fp, fn, sel)
        agree_by_cohort[name] = _agree_sel(agree, sel)
        lm.log(f"val/cohort/{name}/val_dice", dice_by_cohort[name], **d)
        lm.log(f"val/cohort/{name}/val_prompt_agreement", agree_by_cohort[name], **d)
    lm.log("val_dice_weighted", _weighted_mean(dice_by_cohort, cohort_weights), **d)
    lm.log("val_prompt_agreement_weighted", _weighted_mean(agree_by_cohort, cohort_weights), **d)

    # (f) tags
    for name, val in (("click_inside", 1), ("click_outside", 0)):
        lm.log(f"val/tag/{name}/val_dice", _dice_sel(tp, fp, fn, click_in == val), **d)
    for bi, name in enumerate(SIZE_BUCKETS):
        sel = size_bucket == bi
        lm.log(f"val/tag/{name}/val_dice", _dice_sel(tp, fp, fn, sel), **d)
        lm.log(f"val/tag/{name}/n", float(sel.sum()), reduce_fx="sum", sync_dist=True)
