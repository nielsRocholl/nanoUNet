"""Dense BL-FU pair tensors: full edge index, relational features, row labels."""

from __future__ import annotations

import torch

CROSS_DIM = 11


def dense_pair_index(n_bl: int, n_fu: int, dev=None) -> torch.Tensor:
    row = torch.arange(n_bl, device=dev).repeat_interleave(n_fu)
    col = torch.arange(n_fu, device=dev).repeat(n_bl)
    return torch.stack([row, col], dim=0)


def cross_attr(
    bl_pos: torch.Tensor,
    fu_pos: torch.Tensor,
    bl_x: torch.Tensor,
    fu_x: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    i, j = edge_index
    dp = fu_pos[j] - bl_pos[i]
    dist = torch.linalg.norm(dp, dim=1, keepdim=True)
    bl_st, fu_st = bl_x[i, 1372:1375], fu_x[j, 1372:1375]
    dst = fu_st - bl_st
    same_type = (bl_x[i, 1375].long() == fu_x[j, 1375].long()).float().unsqueeze(1)
    return torch.cat(
        [
            dp / 100.0,
            dist / 100.0,
            torch.log1p(dist) / 5.0,
            dst[:, 0:1],
            dst[:, 0:1].abs(),
            dst[:, 1:2],
            dst[:, 1:2].abs(),
            dst[:, 2:3],
            same_type,
        ],
        dim=1,
    ).to(torch.float32)


def reverse_cross_attr(attr: torch.Tensor) -> torch.Tensor:
    out = attr.clone()
    out[:, 0:3] = -out[:, 0:3]
    out[:, 5:6] = -out[:, 5:6]
    out[:, 7:8] = -out[:, 7:8]
    out[:, 9:10] = -out[:, 9:10]
    return out
