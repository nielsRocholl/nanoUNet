"""MAE encoder ROI crop + masked mean pool for lesion node features (320-D default).

On-the-fly CT resample + CTNormalization approximates nnUNet preprocess; not bit-identical to .b2nd.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import load_json
from scipy.ndimage import zoom

from nanounet.model.mae_transfer import load_mae_encoder
from nanounet.model.network import build_net
from nanounet.plan.plans import Plans, determine_num_input_channels
from tracking.data.features import FeatConfig, MAE_DIM


class MaeExtractor:
    def __init__(self, cfg: FeatConfig, device: str | None = None):
        assert cfg.mode == "mae"
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        plans_path = Path(cfg.mae_plans)
        self.plans = Plans(str(plans_path))
        self.cm = self.plans.get_configuration("3d_fullres")
        dj_path = plans_path.parent / "dataset.json"
        self.dj = load_json(str(dj_path))
        self.lm = self.plans.get_label_manager(self.dj)
        n_in = determine_num_input_channels(self.cm, self.dj)
        net = build_net(self.cm, self.lm, self.dj, enable_deep_supervision=False, n_extra_in=0, num_classes_override=n_in)
        self._load_encoder(net, cfg.mae_ckpt)
        self.encoder = net.encoder.to(self.device).eval()
        self.patch_size = tuple(int(x) for x in self.cm.patch_size)
        self.tgt_sp = np.asarray(self.cm.spacing, dtype=np.float64)
        ch = self.plans.foreground_intensity_properties_per_channel["0"]
        self.clip_lo = float(ch["percentile_00_5"])
        self.clip_hi = float(ch["percentile_99_5"])
        self.ct_mean = float(ch["mean"])
        self.ct_std = float(ch["std"])
        self._vol_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _load_encoder(net: torch.nn.Module, ckpt_path: str) -> None:
        load_mae_encoder(net, ckpt_path)

    def _norm(self, vol: np.ndarray) -> np.ndarray:
        out = vol.astype(np.float32, copy=True)
        np.clip(out, self.clip_lo, self.clip_hi, out=out)
        out -= self.ct_mean
        out /= max(self.ct_std, 1e-8)
        return out

    def _resample_pair(self, ct: np.ndarray, mask: np.ndarray, sp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zf = sp / self.tgt_sp
        if self.device.type == "cuda":
            sh = tuple(int(round(s * z)) for s, z in zip(ct.shape, zf))
            with torch.inference_mode():
                xc = torch.from_numpy(ct[None, None]).to(self.device, non_blocking=True)
                xm = torch.from_numpy(mask[None, None].astype(np.float32)).to(self.device, non_blocking=True)
                rc = F.interpolate(xc, size=sh, mode="trilinear", align_corners=False).squeeze().float().cpu().numpy()
                rm = F.interpolate(xm, size=sh, mode="nearest").squeeze().round().cpu().numpy().astype(np.int32)
            return rc, rm, zf
        rc = zoom(ct, zf, order=1, mode="nearest")
        rm = zoom(mask, zf, order=0, mode="nearest").astype(np.int32)
        return rc, rm, zf

    def _prepare(self, ct: np.ndarray, mask: np.ndarray, sp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = id(ct)
        hit = self._vol_cache.get(key)
        if hit is not None:
            return hit
        rc, rm, zf = self._resample_pair(ct, mask, sp)
        out = (self._norm(rc), rm, zf)
        self._vol_cache[key] = out
        return out

    def _crop_roi(self, vol: np.ndarray, lbl: np.ndarray, lid: int, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pd, ph, pw = self.patch_size
        c = np.round(center).astype(int)
        lo = c - np.array([pd // 2, ph // 2, pw // 2])
        hi = lo + np.array([pd, ph, pw])
        roi = np.zeros(self.patch_size, np.float32)
        mroi = np.zeros(self.patch_size, np.float32)
        vlo = np.maximum(lo, 0)
        vhi = np.minimum(hi, vol.shape)
        rlo = vlo - lo
        rhi = vhi - lo
        roi[rlo[0] : rhi[0], rlo[1] : rhi[1], rlo[2] : rhi[2]] = vol[vlo[0] : vhi[0], vlo[1] : vhi[1], vlo[2] : vhi[2]]
        sl = (slice(rlo[0], rhi[0]), slice(rlo[1], rhi[1]), slice(rlo[2], rhi[2]))
        vs = (slice(vlo[0], vhi[0]), slice(vlo[1], vhi[1]), slice(vlo[2], vhi[2]))
        mroi[sl] = (lbl[vs] == lid).astype(np.float32)
        return roi, mroi

    @torch.inference_mode()
    def _pool_batch(self, patches: torch.Tensor, masks: torch.Tensor) -> np.ndarray:
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
            skips = self.encoder(patches)
        si = self.cfg.mae_skip
        assert 0 <= si < len(skips)
        feat = skips[si].float()
        m = torch.nn.functional.interpolate(masks.unsqueeze(1), size=feat.shape[2:], mode="nearest").squeeze(1)
        w = m.sum(dim=(1, 2, 3)).clamp_min(1e-6)
        pooled = (feat * m.unsqueeze(1)).sum(dim=(2, 3, 4)) / w.unsqueeze(1)
        assert pooled.shape[1] == MAE_DIM
        return pooled.cpu().numpy().astype(np.float32)

    def prepare_rois(
        self,
        ct: np.ndarray,
        mask: np.ndarray,
        sp: np.ndarray,
        lesion_ids: list[int],
        centers: dict[int, np.ndarray],
    ) -> tuple[list[int], np.ndarray, np.ndarray]:
        if not lesion_ids:
            return [], np.empty((0, *self.patch_size), np.float32), np.empty((0, *self.patch_size), np.float32)
        rv, rm, zf = self._prepare(ct, mask, sp)
        rois, mrois, order = [], [], []
        for lid in lesion_ids:
            roi, mroi = self._crop_roi(rv, rm, lid, centers[lid] * zf)
            rois.append(roi)
            mrois.append(mroi)
            order.append(lid)
        return order, np.stack(rois), np.stack(mrois)

    def infer_rois(self, order: list[int], rois: np.ndarray, mrois: np.ndarray) -> dict[int, np.ndarray]:
        if not order:
            return {}
        out: dict[int, np.ndarray] = {}
        pin = self.device.type == "cuda"
        bs = max(1, self.cfg.mae_batch)
        for i in range(0, len(order), bs):
            sl = slice(i, i + bs)
            x = torch.from_numpy(rois[sl]).unsqueeze(1).to(self.device, non_blocking=pin)
            m = torch.from_numpy(mrois[sl]).to(self.device, non_blocking=pin)
            vecs = self._pool_batch(x, m)
            del x, m
            if pin:
                torch.cuda.empty_cache()
            for j, lid in enumerate(order[sl]):
                out[lid] = vecs[j]
        return out

    def pool_lesions(
        self,
        ct: np.ndarray,
        mask: np.ndarray,
        sp: np.ndarray,
        lesion_ids: list[int],
        centers: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        order, rois, mrois = self.prepare_rois(ct, mask, sp, lesion_ids, centers)
        return self.infer_rois(order, rois, mrois)

    def clear_cache(self) -> None:
        self._vol_cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
