"""Qahqaie et al. (ISBI 2026, arXiv 2602.09933) lesion correspondence as unbalanced entropic optimal transport, reimplemented.

Cost C_ij = c_geom_ij * (1 - w_S * s_ij) with c_geom = ||x_i - x_j|| / (r_i + r_j) (capped) and s_ij the appearance similarity rescaled to
[0, 1]; the paper's registration-trust factor (1 + w_J (1 - tau_ij)) is OMITTED (w_J = 0, owner decision: no deformation field exists for
this data, the paper must say so). Masses are volume fractions, the plan is found by Chizat-style scaling iterations for
min <G, C> + lam KL(G 1 | a) + mu KL(G^T 1 | b) - eps H(G), then relatively pruned and turned into degrees.

Non-obvious choices where the paper is silent (hyperparameters are tuned inside the training folds, never on the scored patients): the
tumour-load prior rho = V_FU / V_BL rescales the marginal penalties (mu_eff = mu min(1, 1/rho), lam_eff = lam min(1, rho)); relative pruning
keeps G_ij if it reaches tau of its row maximum `or`/`and` of its column maximum (`combine` is tuned); because a relative rule always
keeps every row's maximum, an absolute mass floor kappa * min(a_i, b_j) below which an entry is dropped is OUR completion so that disappeared
and new lesions are expressible.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
from __future__ import annotations

import itertools

import numpy as np

CGEOM_CAP = 5.0
ITERS = 200
GRID = {"eps": (0.05, 0.2, 0.5), "lam": (0.1, 1.0), "w_s": (0.0, 0.5), "tau": (0.3, 0.6), "combine": ("or", "and"), "kappa": (0.0, 0.05, 0.2)}


def grid_points() -> list[dict]:
    return [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]


def cost(x_bl: np.ndarray, x_fu: np.ndarray, r_bl: np.ndarray, r_fu: np.ndarray, sim: np.ndarray | None, w_s: float) -> np.ndarray:
    """(n_bl, n_fu) cost; x in mm (BL already in FU space), r equivalent-sphere radii in mm, sim in [0, 1] or None."""
    c = np.linalg.norm(x_bl[:, None, :] - x_fu[None, :, :], axis=-1) / (r_bl[:, None] + r_fu[None, :])
    c = np.minimum(c, CGEOM_CAP)
    return c * (1.0 - w_s * sim) if sim is not None and w_s > 0 else c


def transport(c: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float, lam: float, mu: float) -> np.ndarray:
    """Unbalanced Sinkhorn scaling (Chizat et al. 2018); returns the plan G (n_bl, n_fu)."""
    k = np.exp(-c / eps)
    fl, fm = lam / (lam + eps), mu / (mu + eps)
    u, v = np.ones_like(a), np.ones_like(b)
    for _ in range(ITERS):
        u = (a / np.maximum(k @ v, 1e-300)) ** fl
        v = (b / np.maximum(k.T @ u, 1e-300)) ** fm
    return u[:, None] * k * v[None, :]


def match(x_bl, x_fu, vol_bl, vol_fu, sim, prm: dict) -> list[tuple[int, int]]:
    """Edges (bl index, fu index) for hyperparameters `prm` (keys of GRID)."""
    if len(x_bl) == 0 or len(x_fu) == 0:
        return []
    a, b = vol_bl / vol_bl.sum(), vol_fu / vol_fu.sum()
    rho = vol_fu.sum() / vol_bl.sum()
    radius = lambda v: (3.0 * v / (4.0 * np.pi)) ** (1.0 / 3.0)
    c = cost(x_bl, x_fu, radius(vol_bl), radius(vol_fu), sim, prm["w_s"])
    g = transport(c, a, b, prm["eps"], prm["lam"] * min(1.0, rho), prm["lam"] * min(1.0, 1.0 / rho))
    row = g >= prm["tau"] * g.max(axis=1, keepdims=True)
    col = g >= prm["tau"] * g.max(axis=0, keepdims=True)
    keep = (row | col if prm["combine"] == "or" else row & col) & (g >= prm["kappa"] * np.minimum(a[:, None], b[None, :]))
    return [(int(i), int(j)) for i, j in np.argwhere(keep)]
