"""LightningModule: step-clock Sinkhorn matcher, graph InfoNCE, hard pairs, EMA."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torch_geometric.data import Batch
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from tracking.matcher import Matcher, MatcherOutput, ModelConfig
from tracking.train.match_utils import (
    focal_bce_with_logits,
    graph_val_counts,
    hard_pair_bce,
    infonce_batch,
    infonce_graphs,
    sinkhorn_edge_scores,
    split_per_graph,
)
from tracking.train.sinkhorn import sinkhorn_loss
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
        hard_pair_w: float = 0.0,
        dust_w: float = 0.30,
        dust_pos_w: float = 1.0,
        nce_tau: float = 0.1,
        proj_dim: int = 64,
        sinkhorn_iters: int = 20,
        max_epochs: int = 400,
        max_steps: int = 40000,
        warmup_steps: int = 1000,
        dust_pair_summary: bool = True,
        dust_legacy_linear: bool = False,
        set_attn_blocks: int = 0,
        sinkhorn_uniform_fu: bool = True,
        ema_decay: float = 0.999,
        ema_start_step: int = 1000,
        tta_n: int = 0,
        dust_tau: float = 0.2,
        k_intra: int = 8,
        hard_k: int = 4,
        fu_jitter_scale: float = 0.3,
        desc_jitter_frac: float = 0.0,
        desc_dim: int = 1372,
        desc_norm: bool = False,
        nce_scope: str = "graph",
        edge_cross_attn: bool = False,
        val_score_ema_beta: float = 0.3,
    ):
        super().__init__()
        assert nce_scope in ("graph", "batch")
        self.save_hyperparameters()
        mc = ModelConfig(d=d, layers=layers, heads=heads, dropout=dropout, desc_dim=desc_dim, use_dust_pair_summary=dust_pair_summary, dust_legacy_linear=dust_legacy_linear, set_attn_blocks=set_attn_blocks, desc_norm=desc_norm, edge_cross_attn=edge_cross_attn, sinkhorn_iters=sinkhorn_iters)
        self.matcher = Matcher(mc)
        self._val_score_ewma = None
        self._val_score_peak = self._best_ema_score = self._best_raw_score = 0.0
        self._best_sub: dict[str, float] = {}
        self.ema_matcher = AveragedModel(self.matcher, avg_fn=lambda a, m, n: ema_decay * a + (1.0 - ema_decay) * m) if ema_decay > 0.0 else None
        self.proj = nn.Linear(d, proj_dim)
        self.auroc = BinaryAUROC()
        self.ap_sinkhorn = BinaryAveragePrecision()

    def _ema_ready(self) -> bool:
        return self.ema_matcher is not None and int(self.global_step) >= int(self.hparams.ema_start_step)

    def _val_net(self) -> nn.Module | AveragedModel:
        return self.ema_matcher if self._ema_ready() else self.matcher

    def _forward_val_metrics(self, batch: Batch) -> MatcherOutput:
        net = self._val_net()
        ntta, k, jit, dj = int(self.hparams.tta_n), int(self.hparams.k_intra), float(self.hparams.fu_jitter_scale), float(self.hparams.desc_jitter_frac)
        return net(batch) if ntta <= 0 else matcher_tta_forward(net, batch, ntta, k, jit, dj)

    def forward(self, batch: Batch) -> MatcherOutput:
        return self.matcher(batch)

    def _dust_weight(self) -> float:
        if int(self.hparams.max_steps) > 0:
            ov = getattr(self, "_dust_ramp_step_override", None)
            step = int(ov) if ov is not None else int(self.global_step)
            return float(self.hparams.dust_w) * min(1.0, step / max(1, int(self.hparams.warmup_steps)))
        ov = getattr(self, "_dust_ramp_epoch_override", None)
        epoch = int(ov) if ov is not None else int(self.current_epoch)
        return float(self.hparams.dust_w) * min(1.0, epoch / 20.0)

    def _loss(self, batch: Batch, out: MatcherOutput) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        labels = batch["bl", "cross", "fu"].edge_label
        graphs, pp, db, df = split_per_graph(batch, out)
        sk = [sinkhorn_loss(p, g["bl"].num_nodes, g["fu"].num_nodes, b, f, g["bl", "cross", "fu"].edge_label, self.hparams.sinkhorn_iters, uniform_fu_targets=bool(self.hparams.sinkhorn_uniform_fu)) for g, p, b, f in zip(graphs, pp, db, df)]
        sk_loss = torch.stack(sk).mean()
        pair_focal = focal_bce_with_logits(out.pair, labels)
        if self.hparams.nce_scope == "graph":
            nce = infonce_graphs(graphs, out.z_bl, out.z_fu, self.proj, self.hparams.nce_tau)
        else:
            nce = infonce_batch(out.z_bl, out.z_fu, self.proj, batch["bl", "cross", "fu"].edge_index, labels, self.hparams.nce_tau, batch)
        hard_pair = hard_pair_bce(graphs, pp, int(self.hparams.hard_k)) if float(self.hparams.hard_pair_w) > 0.0 else out.pair.sum() * 0.0
        pw = torch.as_tensor(self.hparams.dust_pos_w, device=out.pair.device, dtype=out.pair.dtype)
        dust_bce = 0.5 * (
            F.binary_cross_entropy_with_logits(out.dust_bl, batch["bl"].no_match_label.to(out.pair.dtype), pos_weight=pw)
            + F.binary_cross_entropy_with_logits(out.dust_fu, batch["fu"].no_match_label.to(out.pair.dtype), pos_weight=pw)
        )
        total = self.hparams.sinkhorn_w * sk_loss + self.hparams.pair_w * pair_focal + self.hparams.nce_w * nce
        total = total + self.hparams.hard_pair_w * hard_pair + self._dust_weight() * dust_bce
        return total, {"sinkhorn_loss": sk_loss, "pair_loss": pair_focal, "nce_loss": nce, "hard_pair_loss": hard_pair, "dust_bce": dust_bce}

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self._ema_ready():
            self.ema_matcher.update_parameters(self.matcher)

    def on_validation_start(self) -> None:
        self._uc_ok = self._uc_tot = self._dis_ok = self._dis_tot = self._new_ok = self._new_tot = 0
    def training_step(self, batch: Batch, _) -> torch.Tensor:
        loss, parts = self._loss(batch, self.matcher(batch))
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs, on_step=True, on_epoch=True)
        for k, v in parts.items():
            self.log(f"train_{k}", v, batch_size=batch.num_graphs, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Batch, _) -> torch.Tensor:
        loss, parts = self._loss(batch, self._val_net()(batch))
        out = self._forward_val_metrics(batch)
        labels = batch["bl", "cross", "fu"].edge_label
        self.auroc.update(torch.sigmoid(out.pair.detach()), labels.int())
        graphs, pp, db, df = split_per_graph(batch, out)
        rows = []
        for g, p, b, f in zip(graphs, pp, db, df):
            self.ap_sinkhorn.update(sinkhorn_edge_scores(g, p.detach(), b.detach(), f.detach(), self.hparams.sinkhorn_iters), g["bl", "cross", "fu"].edge_label.int())
            row, uo, ut, do, dt, no, nt = graph_val_counts(g, p.detach(), b.detach(), f.detach(), self.hparams.sinkhorn_iters, float(self.hparams.dust_tau))
            rows.append(row)
            self._uc_ok += uo
            self._uc_tot += ut
            self._dis_ok += do
            self._dis_tot += dt
            self._new_ok += no
            self._new_tot += nt
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("val_row_acc_hungarian", sum(rows) / len(rows), prog_bar=True, batch_size=batch.num_graphs)
        for k, v in parts.items():
            self.log(f"val_{k}", v, batch_size=batch.num_graphs)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_auroc", self.auroc.compute(), prog_bar=True)
        self.log("val_ap_sinkhorn", self.ap_sinkhorn.compute(), prog_bar=True)
        subs = (("unchanged_split", self._uc_ok, self._uc_tot, 0.5), ("disappeared", self._dis_ok, self._dis_tot, 0.25), ("newly_appearing", self._new_ok, self._new_tot, 0.25))
        parts = []
        for name, ok, tot, w in subs:
            if tot:
                acc = ok / tot
                self.log(f"val_acc_{name}", acc)
                parts.append((w, acc))
        if parts:
            raw = sum(w * a for w, a in parts) / sum(w for w, _ in parts)
            self.log("val_match_score", raw, prog_bar=True)
            b = float(self.hparams.val_score_ema_beta)
            self._val_score_ewma = raw if self._val_score_ewma is None else (1.0 - b) * self._val_score_ewma + b * raw
            self.log("val_match_score_ema", self._val_score_ewma, prog_bar=True)
            self._val_score_peak = max(self._val_score_peak, raw)
            self.log("val_match_score_peak", self._val_score_peak)
            if self._val_score_ewma > self._best_ema_score:
                self._best_ema_score, self._best_raw_score = self._val_score_ewma, raw
                self._best_sub = {n: (o / t if t else 0.0) for n, o, t, _ in subs}
            elif raw > self._best_raw_score:
                self._best_raw_score = raw
        self.auroc.reset()
        self.ap_sinkhorn.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if int(self.hparams.max_steps) > 0:
            ms = int(self.hparams.max_steps)
            warm_steps = min(max(1, int(self.hparams.warmup_steps)), max(1, ms - 1))
            warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warm_steps)
            cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, ms - warm_steps))
            sch = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[warm_steps])
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}
        me = max(1, int(self.hparams.max_epochs))
        sch = torch.optim.lr_scheduler.SequentialLR(opt, [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=5), torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, me - 5))], milestones=[5])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    def predict_batch(self, batch: Batch, tta_n: int | None = None, use_ema: bool = True) -> MatcherOutput:
        net = self.ema_matcher if (use_ema and self.ema_matcher is not None) else self.matcher
        ntta = int(self.hparams.tta_n if tta_n is None else tta_n)
        return net(batch) if ntta <= 0 else matcher_tta_forward(net, batch, ntta, int(self.hparams.k_intra), float(self.hparams.fu_jitter_scale), float(self.hparams.desc_jitter_frac))
