"""LightningModule: Sinkhorn assignment loss, focal pair aux, batch InfoNCE, edge metrics."""

from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, HeteroData
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from tracking.data.graph import FEAT_DIM
from tracking.matcher import Matcher, MatcherOutput, ModelConfig, decode_sinkhorn_hungarian
from tracking.train.sinkhorn import log_sinkhorn, sinkhorn_loss, superglue_marginals

LT_IDX = FEAT_DIM - 1


def focal_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    w = (alpha * target + (1 - alpha) * (1 - target)) * (1 - pt).clamp_min(1e-6).pow(gamma)
    return (w * bce).mean()


def _split_per_graph(
    batch: Batch, out: MatcherOutput
) -> tuple[list[HeteroData], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    graphs = batch.to_data_list()
    es = [g["bl", "cross", "fu"].num_edges for g in graphs]
    nb = [g["bl"].num_nodes for g in graphs]
    nf = [g["fu"].num_nodes for g in graphs]
    pp, db, df = torch.split(out.pair, es), torch.split(out.dust_bl, nb), torch.split(out.dust_fu, nf)
    return graphs, list(pp), list(db), list(df)


def infonce_batch(
    z_bl: torch.Tensor,
    z_fu: torch.Tensor,
    proj: nn.Module,
    edge_index: torch.Tensor,
    edge_label: torch.Tensor,
    tau: float,
    batch: Batch,
) -> torch.Tensor:
    pb = F.normalize(proj(z_bl), dim=1)
    pf = F.normalize(proj(z_fu), dim=1)
    sim = (pb @ pf.T) / tau
    pos = edge_label > 0.5
    if not pos.any():
        return z_bl.sum() * 0.0
    lt_bl = batch["bl"].x[:, LT_IDX].long()
    lt_fu = batch["fu"].x[:, LT_IDX].long()
    same = lt_bl[:, None] == lt_fu[None, :]
    bi, fj = edge_index[0, pos], edge_index[1, pos]
    sort_idx = torch.argsort(bi)
    sb, sf = bi[sort_idx], fj[sort_idx]
    mask = torch.ones(sb.shape[0], dtype=torch.bool, device=sb.device)
    mask[1:] = sb[1:] != sb[:-1]
    bi_u, fj_u = sb[mask], sf[mask]
    sim_a = sim.clone()
    for b in bi_u:
        if same[b].sum() > 1:
            sim_a[b] = sim_a[b].masked_fill(~same[b], float("-inf"))
    la = F.cross_entropy(sim_a[bi_u], fj_u)
    sort_idx = torch.argsort(fj)
    sb, si = fj[sort_idx], bi[sort_idx]
    mask = torch.ones(sb.shape[0], dtype=torch.bool, device=sb.device)
    mask[1:] = sb[1:] != sb[:-1]
    fj_u2, bi_u2 = sb[mask], si[mask]
    sim_t = sim_a.T.clone()
    for f in torch.unique(fj_u2):
        if same[:, f].sum() > 1:
            sim_t[f] = sim_t[f].masked_fill(~same[:, f], float("-inf"))
    lb = F.cross_entropy(sim_t[fj_u2], bi_u2)
    return 0.5 * (la + lb)


def row_hungarian_match_acc(
    data: HeteroData,
    pair_log: torch.Tensor,
    dust_bl: torch.Tensor,
    dust_fu: torch.Tensor,
    iters: int,
) -> float:
    n_bl, n_fu = data["bl"].num_nodes, data["fu"].num_nodes
    lab = data["bl", "cross", "fu"].edge_label.reshape(n_bl, n_fu)
    dec = decode_sinkhorn_hungarian(pair_log, dust_bl, dust_fu, n_bl, n_fu, iters=iters)
    ok = 0
    for i in range(n_bl):
        pos = torch.where(lab[i] > 0.5)[0]
        di = int(dec[i])
        ok += int(di < 0) if pos.numel() == 0 else int((pos == di).any().item())
    return ok / max(n_bl, 1)


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
        dust_w: float = 0.3,
        dust_pos_w: float = 1.0,
        nce_tau: float = 0.1,
        proj_dim: int = 64,
        sinkhorn_iters: int = 20,
        max_epochs: int = 200,
        dust_pair_summary: bool = True,
        dust_legacy_linear: bool = False,
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
            )
        )
        self.proj = nn.Linear(d, proj_dim)
        self.auroc = BinaryAUROC()
        self.ap_sinkhorn = BinaryAveragePrecision()

    def forward(self, batch: Batch) -> MatcherOutput:
        return self.matcher(batch)

    def _loss(self, batch: Batch, out: MatcherOutput) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        labels = batch["bl", "cross", "fu"].edge_label
        pair_focal = focal_bce_with_logits(out.pair, labels)
        graphs, pp, db, df = _split_per_graph(batch, out)
        sk = [
            sinkhorn_loss(p, g["bl"].num_nodes, g["fu"].num_nodes, b, f, g["bl", "cross", "fu"].edge_label, self.hparams.sinkhorn_iters)
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
        out = self.matcher(batch)
        loss, parts = self._loss(batch, out)
        labels = batch["bl", "cross", "fu"].edge_label
        self.auroc.update(torch.sigmoid(out.pair.detach()), labels.int())
        graphs, pp, db, df = _split_per_graph(batch, out)
        it = self.hparams.sinkhorn_iters
        acc = sum(row_hungarian_match_acc(g, p.detach(), b.detach(), f.detach(), it) for g, p, b, f in zip(graphs, pp, db, df)) / len(
            graphs
        )
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
            dec = decode_sinkhorn_hungarian(p.detach(), b_bl.detach(), d_fu.detach(), n_bl, n_fu, iters=it)
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
