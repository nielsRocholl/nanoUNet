"""LightningModule: Sinkhorn assignment loss, focal pair aux, batch InfoNCE, edge metrics."""

from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torch_geometric.data import Batch
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from tracking.decode import decode_sinkhorn_hungarian
from tracking.matcher import Matcher, MatcherOutput, ModelConfig
from tracking.train.match_utils import focal_bce_with_logits, infonce_batch, row_hungarian_match_acc, split_per_graph
from tracking.train.sinkhorn import log_sinkhorn, sinkhorn_loss, superglue_marginals
from tracking.train.tta import matcher_tta_forward


class MatcherModule(pl.LightningModule):
    def __init__(
        self,
        d: int = 128,
        layers: int = 4,
        heads: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        dropout: float = 0.2,
        sinkhorn_w: float = 1.0,
        pair_w: float = 0.1,
        nce_w: float = 0.3,
        dust_w: float = 0.30,
        dust_pos_w: float = 1.0,
        nce_tau: float = 0.1,
        proj_dim: int = 64,
        sinkhorn_iters: int = 20,
        max_epochs: int = 400,
        dust_pair_summary: bool = True,
        dust_legacy_linear: bool = False,
        set_attn_blocks: int = 0,
        sinkhorn_uniform_fu: bool = True,
        ema_decay: float = 0.999,
        ema_start_epoch: int = 5,
        tta_n: int = 0,
        dust_tau: float = 0.2,
        k_intra: int = 8,
        fu_jitter_scale: float = 0.3,
        desc_jitter_frac: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.matcher = Matcher(
            ModelConfig(
                d=d,
                layers=layers,
                heads=heads,
                dropout=dropout,
                use_dust_pair_summary=dust_pair_summary,
                dust_legacy_linear=dust_legacy_linear,
                set_attn_blocks=set_attn_blocks,
            )
        )
        decay = ema_decay
        if decay > 0.0:
            self.ema_matcher: AveragedModel | None = AveragedModel(
                self.matcher,
                avg_fn=lambda a, m, n_avg: decay * a + (1.0 - decay) * m,
            )
        else:
            self.ema_matcher = None
        self.proj = nn.Linear(d, proj_dim)
        self.auroc = BinaryAUROC()
        self.ap_sinkhorn = BinaryAveragePrecision()

    def _val_net(self) -> nn.Module | AveragedModel:
        if self.ema_matcher is None:
            return self.matcher
        if self.current_epoch >= self.hparams.ema_start_epoch:
            return self.ema_matcher
        return self.matcher

    def _forward_val_loss(self, batch: Batch) -> MatcherOutput:
        return self._val_net()(batch)

    def _forward_val_metrics(self, batch: Batch) -> MatcherOutput:
        net = self._val_net()
        if int(self.hparams.tta_n) > 0:
            return matcher_tta_forward(
                net,
                batch,
                int(self.hparams.tta_n),
                int(self.hparams.k_intra),
                float(self.hparams.fu_jitter_scale),
                float(self.hparams.desc_jitter_frac),
            )
        return net(batch)

    def forward(self, batch: Batch) -> MatcherOutput:
        return self.matcher(batch)

    def _loss(self, batch: Batch, out: MatcherOutput) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        labels = batch["bl", "cross", "fu"].edge_label
        pair_focal = focal_bce_with_logits(out.pair, labels)
        graphs, pp, db, df = split_per_graph(batch, out)
        ufu = bool(self.hparams.sinkhorn_uniform_fu)
        sk = [
            sinkhorn_loss(
                p,
                g["bl"].num_nodes,
                g["fu"].num_nodes,
                b,
                f,
                g["bl", "cross", "fu"].edge_label,
                self.hparams.sinkhorn_iters,
                uniform_fu_targets=ufu,
            )
            for g, p, b, f in zip(graphs, pp, db, df)
        ]
        sk_loss = torch.stack(sk).mean()
        nce = infonce_batch(out.z_bl, out.z_fu, self.proj, batch["bl", "cross", "fu"].edge_index, labels, self.hparams.nce_tau, batch)
        pw = torch.as_tensor(self.hparams.dust_pos_w, device=out.pair.device, dtype=out.pair.dtype)
        tgt_b = batch["bl"].no_match_label.to(out.pair.dtype)
        tgt_f = batch["fu"].no_match_label.to(out.pair.dtype)
        dust_bce = 0.5 * (
            F.binary_cross_entropy_with_logits(out.dust_bl, tgt_b, pos_weight=pw)
            + F.binary_cross_entropy_with_logits(out.dust_fu, tgt_f, pos_weight=pw)
        )
        dust_w_eff = self.hparams.dust_w * min(1.0, float(self.current_epoch) / 20.0)
        total = (
            self.hparams.sinkhorn_w * sk_loss
            + self.hparams.pair_w * pair_focal
            + self.hparams.nce_w * nce
            + dust_w_eff * dust_bce
        )
        return total, {"sinkhorn_loss": sk_loss, "pair_loss": pair_focal, "nce_loss": nce, "dust_bce": dust_bce}

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.ema_matcher is not None:
            self.ema_matcher.update_parameters(self.matcher)

    def on_validation_start(self) -> None:
        self._uc_ok = self._uc_tot = 0
        self._dis_ok = self._dis_tot = 0
        self._new_ok = self._new_tot = 0

    def training_step(self, batch: Batch, _) -> torch.Tensor:
        out = self.matcher(batch)
        loss, parts = self._loss(batch, out)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs, on_step=True, on_epoch=True)
        for k, v in parts.items():
            self.log(f"train_{k}", v, batch_size=batch.num_graphs, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Batch, _) -> torch.Tensor:
        out_m = self._forward_val_loss(batch)
        loss, parts = self._loss(batch, out_m)
        out = self._forward_val_metrics(batch)
        labels = batch["bl", "cross", "fu"].edge_label
        self.auroc.update(torch.sigmoid(out.pair.detach()), labels.int())
        graphs, pp, db, df = split_per_graph(batch, out)
        it = self.hparams.sinkhorn_iters
        tau = float(self.hparams.dust_tau)
        acc = sum(
            row_hungarian_match_acc(g, p.detach(), b.detach(), f.detach(), it, tau) for g, p, b, f in zip(graphs, pp, db, df)
        ) / len(graphs)
        for g, p, b_bl, d_fu in zip(graphs, pp, db, df):
            n_bl, n_fu = g["bl"].num_nodes, g["fu"].num_nodes
            dev, dt = p.device, p.dtype
            S = torch.zeros((n_bl + 1, n_fu + 1), device=dev, dtype=dt)
            S[:n_bl, :n_fu] = p.reshape(n_bl, n_fu)
            S[:n_bl, n_fu] = b_bl.detach()
            S[n_bl, :n_fu] = d_fu.detach()
            la, lb = superglue_marginals(n_bl, n_fu, dev, dt)
            P = log_sinkhorn(S, it, la, lb).exp()
            Rn = P[:n_bl] / P[:n_bl].sum(dim=1, keepdim=True).clamp_min(1e-9)
            ei = g["bl", "cross", "fu"].edge_index
            self.ap_sinkhorn.update(Rn[ei[0], ei[1]].detach(), g["bl", "cross", "fu"].edge_label.int())

            lab = g["bl", "cross", "fu"].edge_label.reshape(n_bl, n_fu).cpu().numpy()
            dec = decode_sinkhorn_hungarian(p.detach(), b_bl.detach(), d_fu.detach(), n_bl, n_fu, iters=it, tau=tau)
            no_bl = g["bl"].no_match_label.cpu().numpy()
            no_fu = g["fu"].no_match_label.cpu().numpy()
            for i in range(n_bl):
                if lab[i].any():
                    js = set(map(int, np.where(lab[i] > 0.5)[0]))
                    di = int(dec[i])
                    self._uc_tot += 1
                    self._uc_ok += int(di in js)
            for i in range(n_bl):
                if no_bl[i] > 0.5:
                    self._dis_tot += 1
                    self._dis_ok += int(int(dec[i]) < 0)
            claimed = {int(dec[i]) for i in range(n_bl) if int(dec[i]) >= 0}
            for j in range(n_fu):
                if no_fu[j] > 0.5:
                    self._new_tot += 1
                    self._new_ok += int(j not in claimed)

        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("val_row_acc_hungarian", acc, prog_bar=True, batch_size=batch.num_graphs)
        for k, v in parts.items():
            self.log(f"val_{k}", v, batch_size=batch.num_graphs)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_auroc", self.auroc.compute(), prog_bar=True)
        self.log("val_ap_sinkhorn", self.ap_sinkhorn.compute(), prog_bar=True)
        if self._uc_tot:
            self.log("val_acc_unchanged_split", self._uc_ok / self._uc_tot)
        if self._dis_tot:
            self.log("val_acc_disappeared", self._dis_ok / self._dis_tot)
        if self._new_tot:
            self.log("val_acc_newly_appearing", self._new_ok / self._new_tot)
        parts_ms: list[tuple[float, float]] = []
        if self._uc_tot:
            parts_ms.append((0.5, self._uc_ok / self._uc_tot))
        if self._dis_tot:
            parts_ms.append((0.25, self._dis_ok / self._dis_tot))
        if self._new_tot:
            parts_ms.append((0.25, self._new_ok / self._new_tot))
        ws = sum(w for w, _ in parts_ms)
        if ws > 0:
            self.log("val_match_score", sum(w * a for w, a in parts_ms) / ws, prog_bar=True)
        self.auroc.reset()
        self.ap_sinkhorn.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        me = max(1, int(self.hparams.max_epochs))
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=5)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, me - 5))
        sch = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[5])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    def predict_batch(self, batch: Batch, tta_n: int | None = None, use_ema: bool = True) -> MatcherOutput:
        """Inference: optional EMA + TTA (for CLIs)."""
        ntta = int(self.hparams.tta_n if tta_n is None else tta_n)
        net: nn.Module = self.ema_matcher if (use_ema and self.ema_matcher is not None) else self.matcher
        if ntta > 0:
            return matcher_tta_forward(
                net,
                batch,
                ntta,
                int(self.hparams.k_intra),
                float(self.hparams.fu_jitter_scale),
                float(self.hparams.desc_jitter_frac),
            )
        return net(batch)
