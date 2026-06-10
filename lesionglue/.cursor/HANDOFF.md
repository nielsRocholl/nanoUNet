# Handoff — r9_base frozen, cleanup done

Date: 2026-06-09.

## TL;DR
**Phase:** Round 9 closed. **r9_base** frozen winner (~0.946 `val_match_score_ema`). Major cleanup refactor done — config-driven, L0-only, nanochat-style. **README** current; **technical.md** stale (not updated this session). **No more matcher ablations.**

**Next:** `RUN_FINAL=1 bash scripts/round9.sh` on cluster for honest held-out test. Round 10 pivots to **data** (harder same-anatomy pairs, data quality), not architecture.

---

## Current status

| Area | State |
|------|-------|
| Model | r9_base — L0 descriptors, RowMatchability dustbin, bilinear matcher |
| Config | `configs/base.json` + `tracking/config.py` (`Config`, `load_config`, `dump_config`, `CKPT_MONITOR`) |
| Features | L0 only: `CACHE_TAG=v5_l0`, `DESC_DIM=1372` — no cache rebuild needed |
| R9 CV | 24/25 folds completed before crash; analysis done; all configs within noise |
| Code audit | `cli/`, `data/` clean; refactor goals met |
| Docs | `README.md` updated; `technical.md` significantly outdated |

---

## Frozen / decided

- **Winner:** `r9_base` = values in `configs/base.json` (rollback defaults: `nce_w=0.3`, `dust_w=0.3`, `dust_pos_w=1.0`, etc.)
- **No more matcher knob sweeps** — deltas unmeasurable at current data scale (~0.3pp within fold noise)
- **Single model path:** `module_from_config(cfg)` in `tracking/train/module.py`; no alternate architectures
- **Round 10 direction:** data — more/harder same-anatomy pairs, label/graph quality — **not** matcher tweaks

---

## Cleanup refactor (done)

**Added**
- `tracking/config.py`, `configs/base.json`
- `module_from_config(cfg)` — sole entry to `MatcherModule`
- CLIs `train`, `cv`, `report` take `--config`; training dumps `{out}/config.json`
- `scripts/round9.sh`, `scripts/lesion-round9-cv.sh`
- `predict_masks.py` has `__main__` guard

**Deleted**
- `tracking/set_attn.py`, `tracking/data/mae.py`, `tracking/train/tta.py`
- `scripts/profile_mae.py`
- `FeatConfig`, `add_feat_args`, old `TrainConfig` flag soup, `hard_pair` path

**Unchanged invariants**
- Preprocess cache tag `v5_l0` — existing `.pt` files still valid
- Step-clock training, EMA-by-update-count, patient-level 5-fold CV

---

## R9 CV summary (reference)

- **Best:** `r9_base` ~**0.946** mean `val_match_score_ema`
- **Fold identity dominates** variance; config gaps ~**0.3pp** — within noise
- Rollback knobs (`nce_w` flat 0.1–0.3, dust weights) confirmed safe
- Full analysis in W&B project `lesion-tracking`; judge **peaks not finals**

Configs compared (all tied within noise): `r9_base` plus ablations — no further sweeps planned.

---

## How to run

Repo root, `PYTHONPATH=.`, interpreter `python3` (no `python` on cluster).

```bash
# preprocess (once; skip if cache exists)
python3 tracking/cli/preprocess.py --split all --jobs 4

# train
python3 tracking/cli/train.py --config configs/base.json --out runs/my_run --wandb

# 5-fold CV
python3 tracking/cli/cv.py --config configs/base.json --out runs/cv --wandb

# eval / predict
python3 tracking/cli/eval.py --ckpt runs/my_run/best.ckpt --split val
python3 tracking/cli/predict.py --ckpt runs/my_run/best.ckpt --split val --out preds

# report (train or --checkpoint)
python3 tracking/cli/report.py --config configs/base.json --out runs/report
```

**Round 9 pipeline** (`scripts/round9.sh`):
```bash
export RUNS=runs/round9          # default
export CONFIG=configs/base.json  # default
bash scripts/round9.sh           # CV only; skips if cv_summary.json exists
RUN_FINAL=1 bash scripts/round9.sh   # retrain → dust_tau sweep on val → test ONCE
```

**Cluster:** `scripts/lesion-round9-cv.sh` — SLURM, `RUNS=/nnunet_data/lesion_tracking/runs/round9`.

Env overrides: `RUNS`, `CONFIG`, `WANDB_PROJECT` (default `lesion-tracking`).

---

## Carry-overs (never drop)

1. **Batch-size independence:** step clock + EMA-by-update-count (R8 keeper)
2. **Patient-level 5-fold CV:** mean±std CIs; monitor `val_match_score_ema` (`CKPT_MONITOR`)
3. **Peaks not finals:** heavy overfitting — best ≠ last logged step
4. **Test split touched once:** only via `RUN_FINAL=1` in `scripts/round9.sh`
5. **nanochat-style:** read `.cursor/rules/nanochat-style.mdc` before editing code

---

## Pending

| Priority | Action | Notes |
|----------|--------|-------|
| **GPU** | `RUN_FINAL=1 bash scripts/round9.sh` on cluster | Honest held-out test number; **not run locally yet** |
| Optional | Rewrite `technical.md` | Audit found it stale; README is source for pipeline |
| Round 10 | Data-focused plan | Harder same-anatomy pairs, QC — not matcher architecture |

---

## Key paths

| Path | Role |
|------|------|
| `configs/base.json` | Canonical r9_base hyperparams |
| `tracking/config.py` | `Config` dataclass, load/dump, `CKPT_MONITOR` |
| `tracking/train/module.py` | `module_from_config`, `MatcherModule` |
| `tracking/data/features.py` | `DESC_DIM`, `CACHE_TAG` |
| `tracking/cli/{train,cv,report,eval,predict}.py` | Pipeline entry points |
| `scripts/round9.sh` | CV → optional final + tau sweep + test gate |
| `scripts/lesion-round9-cv.sh` | Cluster SLURM wrapper |
| `.cursor/rules/nanochat-style.mdc` | Coding philosophy (HARD RULE) |
| `README.md` | Current pipeline docs |

---

## Repo hygiene

- Uncommitted changes may exist from cleanup refactor — **do not commit unless asked**
- W&B: project `lesion-tracking` (or `hyper-alignment/lesion-tracking` in MCP)

---

## Do NOT

- Edit `.cursor/plans/` or plan files unless explicitly tasked
- Rerun 10h matcher ablation sweeps — R9 closed that question
- Touch **test** split before `RUN_FINAL=1`
- Reintroduce `FeatConfig`, MAE, TTA, set_attn, or multi-feature modes
- Assume `technical.md` is current — use `README.md` + code

---

## New-agent bootstrap

Use Block A in `.cursor/agent_prompt_template.md`. Read this file first, then nanochat rules. Confirm state before changing anything.
