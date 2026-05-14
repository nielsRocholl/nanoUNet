"""Checkpoint load, preprocess case, single-patch (+optional border BFS) logits."""

from __future__ import annotations

import os
from collections import deque
from typing import List, Sequence, Tuple

import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nanounet.config import RoiPromptConfig, load_config
from nanounet.infer.border_expand import plan_border_expansion_centers_from_logits
from nanounet.infer.gaussian import gaussian_tile
from nanounet.infer.roi_slices import (
    background_logits_vector,
    centered_spatial_slices_at_point,
    local_prompt_points_for_patch,
    map_points_zyx_unpadded_to_padded,
    safe_divide_merged_logits,
    spatial_slices_to_tuple,
)
from nanounet.infer.tta import predict_with_optional_tta
from nanounet.model.network import build_net
from nanounet.plan.case_pp import run_case
from nanounet.plan.labels import labels_from_dataset_json
from nanounet.plan.plans import Plans
from nanounet.prompt.encoding import encode_points_to_heatmap_pair


def _strip_pl_state(sd: dict) -> dict:
    return {k[4:]: v for k, v in sd.items() if k.startswith("net.")}


def _acc_dtype(dev: torch.device) -> torch.dtype:
    if dev.type == "cpu":
        return torch.float32
    r = (os.environ.get("NNUNET_SINGLE_PATCH_ACCUM_DTYPE") or "").lower()
    if r in ("half", "float16", "fp16"):
        return torch.float16
    return torch.float32


def _fill_workon(
    workon: torch.Tensor,
    data_pad: torch.Tensor,
    sz: slice,
    sy: slice,
    sx: slice,
    n_img: int,
    encode: bool,
    pos: List[Tuple[int, int, int]],
    neg: List[Tuple[int, int, int]],
    cfg: RoiPromptConfig,
    patch: Tuple[int, int, int],
    dev: torch.device,
) -> None:
    sl = (slice(None), sz, sy, sx)
    workon[0, :n_img].copy_(data_pad[sl])
    if not encode:
        workon[0, n_img:].zero_()
        return
    pr = encode_points_to_heatmap_pair(
        pos,
        neg,
        patch,
        cfg.prompt.point_radius_vox,
        cfg.prompt.encoding,
        device=dev,
        intensity_scale=cfg.prompt.prompt_intensity_scale,
    )
    workon[0, n_img:].copy_(pr.float())


def load_net_from_ckpt(ckpt_path: str, cm, dj: dict, dev: torch.device):
    lm = labels_from_dataset_json(dj)
    net = build_net(cm, lm, dj, enable_deep_supervision=False)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    st = _strip_pl_state(sd)
    if not st:
        raise RuntimeError("no net.* keys in checkpoint")
    net.load_state_dict(st, strict=True)
    return net.to(dev).eval(), lm


def pick_checkpoint(model_dir: str, ckpt: str | None) -> str:
    if ckpt:
        if os.path.isfile(ckpt):
            return ckpt
        p2 = join(model_dir, ckpt)
        if os.path.isfile(p2):
            return p2
        raise FileNotFoundError(ckpt)
    cdir = join(model_dir, "checkpoints")
    p = join(cdir, "last.ckpt")
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(f"no checkpoint in {cdir}")


@torch.inference_mode()
def predict_logits_preprocessed(
    image_files: Sequence[str],
    seg_file: str | None,
    model_dir: str,
    ckpt_path: str,
    points_zyx: List[Tuple[int, int, int]],
    roi_cfg: RoiPromptConfig | None = None,
    device: str = "cuda",
    encode_prompt: bool = True,
    border_expand: bool = False,
    max_border_expand_extra: int = 16,
    disable_tta: bool = False,
) -> tuple[torch.Tensor, dict]:
    plans_p = join(model_dir, "plans.json")
    dj = load_json(join(model_dir, "dataset.json"))
    nano_p = join(model_dir, "nano_config.json")
    pl = Plans(plans_p)
    cm = pl.get_configuration("3d_fullres")
    cfg = roi_cfg if roi_cfg is not None else load_config(nano_p)
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    net, lm = load_net_from_ckpt(ckpt_path, cm, dj, dev)
    use_mirroring = not (disable_tta or cfg.inference.disable_tta_default)
    data, _s, props = run_case(list(image_files), seg_file, pl, cm, dj, verbose=False)
    data_t = torch.from_numpy(data).float()
    patch_size = tuple(cm.patch_size)
    pad, slicer_revert = pad_nd_image(data_t, patch_size, "constant", {"value": 0}, True, None)
    padded_shape = tuple(pad.shape[1:])
    pad = pad.to(dev)
    results_dev = dev
    acc_dtype = _acc_dtype(results_dev)
    gaussian = gaussian_tile(patch_size, results_dev, torch.float32)
    gaussian_acc = gaussian.to(dtype=acc_dtype)
    points_pad = map_points_zyx_unpadded_to_padded(points_zyx, slicer_revert)
    pz, py, px = points_pad[0]
    seed = (pz, py, px)
    sz, sy, sx = centered_spatial_slices_at_point(pz, py, px, patch_size, padded_shape)
    bg_vec = background_logits_vector(lm, lm.num_segmentation_heads, results_dev, acc_dtype)
    nh = lm.num_segmentation_heads
    n_img = int(pad.shape[0])
    workon = torch.empty((1, n_img + 2, *patch_size), device=dev, dtype=torch.float32)

    def fwd(w: torch.Tensor) -> torch.Tensor:
        return predict_with_optional_tta(net, w, use_mirroring)[0]

    pos0 = local_prompt_points_for_patch(seed, sz, sy, sx, patch_size)
    _fill_workon(workon, pad, sz, sy, sx, n_img, encode_prompt, pos0, [], cfg, patch_size, dev)
    pred_raw = fwd(workon)
    if pred_raw.device != results_dev:
        pred_raw = pred_raw.to(results_dev)

    if not border_expand:
        merged = pred_raw.to(dtype=acc_dtype)
        logits = bg_vec.view(-1, 1, 1, 1).to(dtype=acc_dtype).expand(nh, *padded_shape).clone()
        logits[:, sz, sy, sx] = merged
        logits = logits[(slice(None), *slicer_revert[1:])]
        return logits.float().cpu(), props

    logits_acc = torch.zeros((nh, *padded_shape), dtype=acc_dtype, device=results_dev)
    n_pred = torch.zeros(padded_shape, dtype=acc_dtype, device=results_dev)
    logits_acc[:, sz, sy, sx] += (pred_raw * gaussian).to(dtype=acc_dtype)
    n_pred[sz, sy, sx] += gaussian_acc
    seed_key = spatial_slices_to_tuple(sz, sy, sx)
    vis = {seed_key}
    q = deque(
        plan_border_expansion_centers_from_logits(
            pred_raw, lm, sz, sy, sx, patch_size, padded_shape, seed_key, max_border_expand_extra, skip_keys=vis
        )
    )
    done = 0
    while q and done < max_border_expand_extra:
        pe, pye, pxe = q.popleft()
        sze, sye, sxe = centered_spatial_slices_at_point(pe, pye, pxe, patch_size, padded_shape)
        ke = spatial_slices_to_tuple(sze, sye, sxe)
        if ke in vis:
            continue
        vis.add(ke)
        pose = local_prompt_points_for_patch(seed, sze, sye, sxe, patch_size)
        _fill_workon(workon, pad, sze, sye, sxe, n_img, encode_prompt, pose, [], cfg, patch_size, dev)
        raw_e = fwd(workon)
        if raw_e.device != results_dev:
            raw_e = raw_e.to(results_dev)
        logits_acc[:, sze, sye, sxe] += (raw_e * gaussian).to(dtype=acc_dtype)
        n_pred[sze, sye, sxe] += gaussian_acc
        done += 1
        bud = max_border_expand_extra - done
        if bud > 0:
            for c in plan_border_expansion_centers_from_logits(
                raw_e, lm, sze, sye, sxe, patch_size, padded_shape, seed_key, bud, skip_keys=vis
            ):
                q.append(c)
    safe_divide_merged_logits(logits_acc, n_pred, bg_vec)
    logits_acc = logits_acc[(slice(None), *slicer_revert[1:])]
    return logits_acc.float().cpu(), props
