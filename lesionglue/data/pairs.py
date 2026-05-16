"""Dense BL-FU pair tensors: full edge index, relational features + descriptor sim."""

from __future__ import annotations

import torch
import torch.nn.functional as F

CROSS_DIM = 27
_DESC = 1372
_LT = 1386
_SCALE_L2 = _DESC**0.5


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
    bl_st, fu_st = bl_x[i, _DESC : _DESC + 3], fu_x[j, _DESC : _DESC + 3]
    dst = fu_st - bl_st
    rad = fu_x[j, _DESC:_LT] - bl_x[i, _DESC:_LT]
    d_bl, d_fu = bl_x[i, :_DESC], fu_x[j, :_DESC]
    desc_cos = F.cosine_similarity(d_bl, d_fu, dim=1, eps=1e-6).unsqueeze(1)
    desc_l2 = (d_bl - d_fu).norm(dim=1, keepdim=True) / _SCALE_L2
    same_type = (bl_x[i, _LT].long() == fu_x[j, _LT].long()).float().unsqueeze(1)
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
            desc_cos,
            desc_l2,
            rad,
        ],
        dim=1,
    ).to(torch.float32)


def reverse_cross_attr(attr: torch.Tensor) -> torch.Tensor:
    out = attr.clone()
    out[:, 0:3] = -out[:, 0:3]
    out[:, 5:6] = -out[:, 5:6]
    out[:, 7:8] = -out[:, 7:8]
    out[:, 9:10] = -out[:, 9:10]
    out[:, 13:27] = -out[:, 13:27]
    return out
