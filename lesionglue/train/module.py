"""LightningModule: dense pair BCE, row dustbin loss, no-match BCE, edge metrics."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from tracking.matcher import Matcher, MatcherOutput, ModelConfig


def row_loss(data: HeteroData, pair_logits: torch.Tensor, bl_none: torch.Tensor) -> torch.Tensor:
    n_bl, n_fu = data["bl"].num_nodes, data["fu"].num_nodes
    lab = data["bl", "cross", "fu"].edge_label.to(pair_logits.device).reshape(n_bl, n_fu)
    scores = torch.cat([pair_logits.reshape(n_bl, n_fu), bl_none.reshape(n_bl, 1)], dim=1)
    pos = lab > 0.5
    counts = pos.sum(dim=1)
    target = torch.zeros_like(scores)
    target[:, :n_fu] = pos.float() / counts.clamp_min(1).unsqueeze(1)
    target[counts == 0, n_fu] = 1.0
    return -(target * F.log_softmax(scores, dim=1)).sum(dim=1).mean()


def row_match_acc(data: HeteroData, pair_logits: torch.Tensor, bl_none: torch.Tensor) -> float:
    n_bl, n_fu = data["bl"].num_nodes, data["fu"].num_nodes
    lab = data["bl", "cross", "fu"].edge_label.reshape(n_bl, n_fu).cpu()
    scores = torch.cat([pair_logits.reshape(n_bl, n_fu).cpu(), bl_none.reshape(n_bl, 1).cpu()], dim=1)
    pred = scores.argmax(dim=1)
    ok = 0
    for i in range(n_bl):
        pos = torch.where(lab[i] > 0.5)[0]
        ok += int(pred[i].item() == n_fu) if pos.numel() == 0 else int((pos == pred[i]).any().item())
    return ok / max(n_bl, 1)


class MatcherModule(pl.LightningModule):
    def __init__(
        self,
        d: int = 128,
        layers: int = 4,
        heads: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        pos_weight: float = 1.0,
        pair_w: float = 1.0,
        row_w: float = 0.5,
        none_w: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.matcher = Matcher(ModelConfig(d=d, layers=layers, heads=heads))
        self.register_buffer("pw", torch.tensor(float(pos_weight)))
        self.auroc = BinaryAUROC()
        self.ap = BinaryAveragePrecision()

    def forward(self, batch: Batch) -> MatcherOutput:
        return self.matcher(batch)

    def _loss(self, batch: Batch, out: MatcherOutput) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        labels = batch["bl", "cross", "fu"].edge_label
        pair = F.binary_cross_entropy_with_logits(out.pair, labels, pos_weight=self.pw)
        graphs = batch.to_data_list()
        edge_sizes = [g["bl", "cross", "fu"].num_edges for g in graphs]
        bl_sizes = [g["bl"].num_nodes for g in graphs]
        rows = [row_loss(g, p, b) for g, p, b in zip(graphs, torch.split(out.pair, edge_sizes), torch.split(out.bl_no_match, bl_sizes))]
        row = torch.stack(rows).mean()
        bl_none = F.binary_cross_entropy_with_logits(out.bl_no_match, batch["bl"].no_match_label.float())
        fu_none = F.binary_cross_entropy_with_logits(out.fu_no_match, batch["fu"].no_match_label.float())
        none = 0.5 * (bl_none + fu_none)
        total = self.hparams.pair_w * pair + self.hparams.row_w * row + self.hparams.none_w * none
        return total, {"pair_loss": pair, "row_loss": row, "none_loss": none}

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
        self.ap.update(torch.sigmoid(out.pair.detach()), labels.int())
        graphs = batch.to_data_list()
        edge_sizes = [g["bl", "cross", "fu"].num_edges for g in graphs]
        bl_sizes = [g["bl"].num_nodes for g in graphs]
        acc = sum(row_match_acc(g, p.detach(), b.detach()) for g, p, b in zip(graphs, torch.split(out.pair, edge_sizes), torch.split(out.bl_no_match, bl_sizes))) / len(graphs)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log("val_row_acc", acc, prog_bar=True, batch_size=batch.num_graphs)
        for k, v in parts.items():
            self.log(f"val_{k}", v, batch_size=batch.num_graphs)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_auroc", self.auroc.compute(), prog_bar=True)
        self.log("val_ap", self.ap.compute(), prog_bar=True)
        self.auroc.reset()
        self.ap.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss", "interval": "epoch"}}
