"""LightningModule: step-clock Sinkhorn matcher, graph InfoNCE, EMA-best val_match_score."""

from __future__ import annotations

import copy
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torch_geometric.data import Batch
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from tracking.config import Config
from tracking.matcher import Matcher, MatcherOutput, ModelConfig
from tracking.train.match_utils import (
    focal_bce_with_logits,
    graph_val_counts,
    infonce_graphs,
    sinkhorn_edge_scores,
    split_per_graph,
)
from tracking.train.sinkhorn import sinkhorn_loss

PROJ_DIM = 64
SWA_BAND = 0.01        # raw val_match_score within 1pp of the running max counts as "on the plateau"
SWA_MIN_UPDATES = 5


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
        sinkhorn_iters: int = 20,
        max_steps: int = 8000,
        warmup_steps: int = 1000,
        ema_decay: float = 0.999,
        ema_start_step: int = 1000,
        dust_tau: float = 0.2,
        val_score_ema_beta: float = 0.3,
        k_intra: int = 8,
        drop_dp: bool = False,
        intra: str = "knn",
        type_mask: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        mc = ModelConfig(
            d=d, layers=layers, heads=heads, dropout=dropout, sinkhorn_iters=sinkhorn_iters,
        )
        self.matcher = Matcher(mc)
        self._val_score_ewma = None
        self._val_score_peak = self._best_ema_score = self._best_raw_score = 0.0
        self._best_sub: dict[str, float] = {}
        self.ema_matcher = AveragedModel(self.matcher, avg_fn=lambda a, m, n: ema_decay * a + (1.0 - ema_decay) * m) if ema_decay > 0.0 else None
        # Held in a list so nn.Module never registers it: keeping SWA out of state_dict is what lets
        # load_from_checkpoint keep working on pre-R12 checkpoints. Built lazily on the first
        # plateau hit, by which point matcher is already on the training device.
        self._swa: list[AveragedModel] = []
        self._swa_updates = 0
        self.proj = nn.Linear(d, PROJ_DIM)
        self.auroc = BinaryAUROC()
        self.ap_sinkhorn = BinaryAveragePrecision()
        self._eval_use_ema: bool | None = None  # not in Lightning state; standalone eval sets it

    def _ema_ready(self) -> bool:
        return self.ema_matcher is not None and int(self.global_step) >= int(self.hparams.ema_start_step)

    def set_eval_weights(self, use_ema: bool) -> None:
        if use_ema and self.ema_matcher is None:
            raise RuntimeError(
                "EMA evaluation requested but this checkpoint has no ema_matcher.\n"
                "Expected a Lightning ckpt trained with ema_decay>0.\n"
                "Fix: lesion_track_eval --ckpt … --no-ema"
            )
        self._eval_use_ema = bool(use_ema)

    def _val_net(self) -> nn.Module | AveragedModel:
        if self._eval_use_ema is not None:
            return self.ema_matcher if self._eval_use_ema else self.matcher
        return self.ema_matcher if self._ema_ready() else self.matcher

    def forward(self, batch: Batch) -> MatcherOutput:
        return self.matcher(batch)

    def _dust_weight(self) -> float:
        step = int(getattr(self, "_dust_ramp_step_override", self.global_step))
        return float(self.hparams.dust_w) * min(1.0, step / max(1, int(self.hparams.warmup_steps)))

    def _loss(self, batch: Batch, out: MatcherOutput) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        labels = batch["bl", "cross", "fu"].edge_label
        graphs, pp, db, df = split_per_graph(batch, out)
        sk = [sinkhorn_loss(p, g["bl"].num_nodes, g["fu"].num_nodes, b, f, g["bl", "cross", "fu"].edge_label, self.hparams.sinkhorn_iters) for g, p, b, f in zip(graphs, pp, db, df)]
        sk_loss = torch.stack(sk).mean()
        pair_focal = focal_bce_with_logits(out.pair, labels)
        nce = infonce_graphs(graphs, out.z_bl, out.z_fu, self.proj, self.hparams.nce_tau)
        pw = torch.as_tensor(self.hparams.dust_pos_w, device=out.pair.device, dtype=out.pair.dtype)
        dust_bce = 0.5 * (
            F.binary_cross_entropy_with_logits(out.dust_bl, batch["bl"].no_match_label.to(out.pair.dtype), pos_weight=pw)
            + F.binary_cross_entropy_with_logits(out.dust_fu, batch["fu"].no_match_label.to(out.pair.dtype), pos_weight=pw)
        )
        total = self.hparams.sinkhorn_w * sk_loss + self.hparams.pair_w * pair_focal + self.hparams.nce_w * nce + self._dust_weight() * dust_bce
        return total, {"sinkhorn_loss": sk_loss, "pair_loss": pair_focal, "nce_loss": nce, "dust_bce": dust_bce}

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self._ema_ready():
            self.ema_matcher.update_parameters(self.matcher)

    def on_validation_start(self) -> None:
        self._uc_ok = self._uc_tot = self._dis_ok = self._dis_tot = self._new_ok = self._new_tot = 0
        self._per_patient: dict[str, dict[str, int]] = {}

    def training_step(self, batch: Batch, _) -> torch.Tensor:
        loss, parts = self._loss(batch, self.matcher(batch))
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs, on_step=True, on_epoch=True)
        for k, v in parts.items():
            self.log(f"train_{k}", v, batch_size=batch.num_graphs, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Batch, _) -> torch.Tensor:
        net = self._val_net()
        out = net(batch)
        loss, parts = self._loss(batch, out)
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
            # Batching may put >1 graph from the same patient in one step (e.g. two BL/FU transitions),
            # so accumulate rather than overwrite.
            pid = str(g.pid)
            acc = self._per_patient.setdefault(pid, {"uc_ok": 0, "uc_tot": 0, "dis_ok": 0, "dis_tot": 0, "new_ok": 0, "new_tot": 0})
            acc["uc_ok"] += uo
            acc["uc_tot"] += ut
            acc["dis_ok"] += do
            acc["dis_tot"] += dt
            acc["new_ok"] += no
            acc["new_tot"] += nt
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
            # unchanged_split converges ~500 steps before disappeared/newly decay, so no single step is
            # optimal for all three; averaging the plateau window captures both.
            if raw >= self._val_score_peak - SWA_BAND:
                if not self._swa:
                    self._swa.append(AveragedModel(self.matcher))
                self._swa[0].update_parameters(self.matcher)
                self._swa_updates += 1
            if self._val_score_ewma > self._best_ema_score:
                self._best_ema_score, self._best_raw_score = self._val_score_ewma, raw
                self._best_sub = {n: (o / t if t else 0.0) for n, o, t, _ in subs}
            elif raw > self._best_raw_score:
                self._best_raw_score = raw
        self.auroc.reset()
        self.ap_sinkhorn.reset()

    def on_train_end(self) -> None:
        if not self._swa:
            return
        if self._swa_updates < SWA_MIN_UPDATES:
            raise RuntimeError(
                f"swa_matcher only saw {self._swa_updates} plateau updates (need >= {SWA_MIN_UPDATES}); "
                "a run that never plateaued is broken, not silently skippable."
            )
        path = Path(self.trainer.default_root_dir) / "swa_plateau.ckpt"
        swa_sd = self._swa[0].module.state_dict()
        matcher_sd = copy.deepcopy(self.matcher.state_dict())
        ema_sd = copy.deepcopy(self.ema_matcher.module.state_dict()) if self.ema_matcher is not None else None
        # overwrite both shadow copies so eval.py picks up SWA weights whether or not it reads the EMA branch
        self.matcher.load_state_dict(swa_sd)
        if self.ema_matcher is not None:
            self.ema_matcher.module.load_state_dict(swa_sd)
        self.trainer.save_checkpoint(str(path))
        self.matcher.load_state_dict(matcher_sd)
        if ema_sd is not None:
            self.ema_matcher.module.load_state_dict(ema_sd)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        ms = int(self.hparams.max_steps)
        warm_steps = min(max(1, int(self.hparams.warmup_steps)), max(1, ms - 1))
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warm_steps)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, ms - warm_steps))
        sch = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[warm_steps])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

    def predict_batch(self, batch: Batch, use_ema: bool = True) -> MatcherOutput:
        net = self.ema_matcher if (use_ema and self.ema_matcher is not None) else self.matcher
        return net(batch)


def module_from_config(cfg: Config) -> MatcherModule:
    return MatcherModule(
        d=cfg.d, layers=cfg.layers, heads=cfg.heads, lr=cfg.lr, weight_decay=cfg.weight_decay,
        dropout=cfg.dropout, sinkhorn_w=cfg.sinkhorn_w, pair_w=cfg.pair_w, nce_w=cfg.nce_w,
        dust_w=cfg.dust_w, dust_pos_w=cfg.dust_pos_w, nce_tau=cfg.nce_tau,
        sinkhorn_iters=cfg.sinkhorn_iters, max_steps=cfg.max_steps, warmup_steps=cfg.warmup_steps,
        ema_decay=cfg.ema_decay, ema_start_step=cfg.ema_start_step, dust_tau=cfg.dust_tau,
        val_score_ema_beta=cfg.val_score_ema_beta,
        k_intra=cfg.k_intra, drop_dp=cfg.drop_dp, intra=cfg.intra, type_mask=cfg.type_mask,
    )
