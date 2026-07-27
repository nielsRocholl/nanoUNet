"""Validation-time Dice metrics, built on the tp/fp/fn core in dice_loss.py: pooled pseudo-Dice,
per-row lesion/no-lesion split, click-inside-vs-outside split, and prompt-pair agreement."""

from __future__ import annotations

import numpy as np
import torch

from nanounet.model.dice_loss import get_tp_fp_fn_tn


def pooled_fg_dice(buf) -> float:
    """nnU-Net global pseudo-Dice: pool per-class fg tp/fp/fn over the whole val buffer."""
    tg = torch.stack([v["tp"] for v in buf]).sum(0).numpy()
    pg = torch.stack([v["fp"] for v in buf]).sum(0).numpy()
    ng = torch.stack([v["fn"] for v in buf]).sum(0).numpy()
    dg = [2 * a / (2 * a + b + c) if (2 * a + b + c) > 0 else np.nan for a, b, c in zip(tg, pg, ng)]
    return float(np.nanmean(dg))


def val_split_metrics(tp, fp, fn, y, output_seg):
    """Split a val batch into the two prompted-test cases by GT foreground presence.

    Returns global per-class fg sums (tp/fp/fn pooled over the batch) plus per-patch
    macro fg Dice on lesion patches (A) and predicted-fg fraction on no-lesion patches
    (B). tp/fp/fn are the foreground slices, shape [B, Cfg].
    """
    has_fg = (y > 0).flatten(1).any(1)
    tps, fps, fns = tp.sum(1), fp.sum(1), fn.sum(1)
    den = 2 * tps + fps + fns
    dice_s = torch.where(den > 0, 2 * tps / den, torch.zeros_like(den))
    pred_fg = (output_seg > 0).float().flatten(1).mean(1)
    return (
        tp.sum(0).detach().cpu(),
        fp.sum(0).detach().cpu(),
        fn.sum(0).detach().cpu(),
        dice_s[has_fg].detach().cpu(),
        pred_fg[~has_fg].detach().cpu(),
    )


def val_step_row(out, y, label_manager, enable_ds: bool, loss_val: float, click_inside=None) -> dict:
    """One validation batch → per-region metric row (tp/fp/fn, macro dice, fp count, loss).

    `click_inside` (optional, -1/0/1 per row -- see patch_render.click_inside_flags) additionally
    splits the has-foreground rows' per-row Dice into `dice_click_in` / `dice_click_out`, feeding
    val_dice_click_inside / val_dice_click_outside (deployment-split diagnostic)."""
    if enable_ds:
        out = out[0]
        y = y[0]
    axes = list(range(2, out.ndim))
    output_seg = out.argmax(1)[:, None]
    oh = torch.zeros_like(out, dtype=torch.float32, device=out.device)
    oh.scatter_(1, output_seg, 1)
    if label_manager.has_ignore_label:
        mask = (y != label_manager.ignore_label).float()
        y = y.clone()
        y[y == label_manager.ignore_label] = 0
    else:
        # Instance-labeled targets (each lesion a distinct id) / out-of-FOV -1 vs a binary
        # head: collapse positives to foreground for a 2-class head, else just drop -1. Keeps
        # the metric one-hot scatter in bounds; no-op for {0,1} data.
        mask = None
        if out.shape[1] == 2:
            y = (y > 0).to(y.dtype)
        else:
            y = y.clamp_min(0)
    tp, fp, fn, _ = get_tp_fp_fn_tn(oh, y, axes=axes, mask=mask)
    tp_fg, fp_fg, fn_fg = tp[:, 1:], fp[:, 1:], fn[:, 1:]
    tg, pg, ng, da, fb = val_split_metrics(tp_fg, fp_fg, fn_fg, y, output_seg)
    row = {"tp": tg, "fp": pg, "fn": ng, "dice_a": da, "fp_b": fb, "loss": loss_val}
    if click_inside is not None:
        has_fg = (y > 0).flatten(1).any(1).cpu()
        ci = click_inside.cpu()
        den = 2 * tp_fg.sum(1) + fp_fg.sum(1) + fn_fg.sum(1)
        dice_row = torch.where(den > 0, 2 * tp_fg.sum(1) / den, torch.zeros_like(den)).cpu()
        valid = has_fg & (ci >= 0)
        row["dice_click_in"] = dice_row[valid & (ci == 1)]
        row["dice_click_out"] = dice_row[valid & (ci == 0)]
    return row


def click_split_means(buf) -> tuple[float, float]:
    """val_dice_click_inside / val_dice_click_outside: mean per-row Dice (see val_step_row's
    dice_click_in/out), nan if a bucket is empty this epoch rather than a fabricated 0/1."""
    din = torch.cat([v["dice_click_in"] for v in buf if "dice_click_in" in v])
    dout = torch.cat([v["dice_click_out"] for v in buf if "dice_click_out" in v])
    return (float(din.mean()) if din.numel() else float("nan")), (float(dout.mean()) if dout.numel() else float("nan"))


def agreement_mean(buf) -> float:
    """val_prompt_agreement: mean of prompt_pair_dice per-row values, skipping NaN
    (both-predictions-empty) rows rather than scoring them 1.0 or 0.0."""
    if not buf:
        return float("nan")
    agree = torch.cat(buf)
    valid = agree[~torch.isnan(agree)]
    return float(valid.mean()) if valid.numel() else float("nan")


def prompt_pair_dice(out, out2, enable_ds: bool) -> torch.Tensor:
    """val_prompt_agreement: per-row foreground Dice between two predictions on the SAME patch
    from two independently-drawn prompts (argmax fg, NOT compared against ground truth). A row
    where both predictions are empty has undefined Dice -- returned as NaN, filtered out by the
    caller before averaging (not scored as 1.0, which would reward agreeing-on-nothing, and not
    0.0, which would penalise a case with no valid comparison)."""
    if enable_ds:
        out, out2 = out[0], out2[0]
    a = out.argmax(1) > 0
    b = out2.argmax(1) > 0
    axes = tuple(range(1, a.ndim))
    inter = (a & b).sum(dim=axes).float()
    denom = a.sum(dim=axes).float() + b.sum(dim=axes).float()
    dice = torch.where(denom > 0, 2 * inter / denom, torch.full_like(denom, float("nan")))
    return dice.detach().cpu()
