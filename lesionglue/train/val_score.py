"""Validation scoring for MatcherModule: per-subset match accounting and the val_match_score signal.

validation_step counts correct matches per subset (unchanged_split / disappeared / newly_appearing),
overall and per patient; on_validation_epoch_end folds them into val_match_score (weights 0.5 / 0.25 /
0.25), tracks its EMA, running peak and best-so-far, and feeds the SWA average. SWA averages the
plateau window (raw score within SWA_BAND of the peak) because unchanged_split converges ~500 steps
before disappeared/newly decay, so no single step is optimal for all three. module.py binds these as
the Lightning hooks.
"""

from __future__ import annotations

import torch
from torch.optim.swa_utils import AveragedModel
from torch_geometric.data import Batch

from lesionglue.train.objective import graph_val_counts, sinkhorn_edge_scores, split_per_graph

SWA_BAND = 0.01        # raw val_match_score within 1pp of the running max counts as "on the plateau"


def on_validation_start(self) -> None:
    self._uc_ok = self._uc_tot = self._dis_ok = self._dis_tot = self._new_ok = self._new_tot = 0
    self._per_patient: dict[str, dict[str, int]] = {}


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
