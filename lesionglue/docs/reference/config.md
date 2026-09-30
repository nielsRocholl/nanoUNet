# Configuration

All training knobs live in JSON, loaded by [`lesionglue/config.py`](../../config.py) (`Config` dataclass, `load_config()`, `dump_config()`).
Inference defaults live in [`lesionglue/common.py`](../../common.py) (`DEPLOYED_CKPT`, `DEPLOYED_DUST_TAU`).
Train recipes: `lesionglue/configs/base.json` (kNN) and `lesionglue/configs/complete.json` (deployed graph recipe).
Train experiments still use `lesionglue/configs/*.json`; the deployed matcher itself needs no config, its hparams come from the checkpoint.

## Config keys

| Config key | Role |
|------------|------|
| `max_steps`, `val_check_steps`, `warmup_steps`, `early_stop_patience` | Step-clock training; checkpoint monitors `val_match_score_ema` |
| `lr`, `weight_decay`, `batch_size`, `val_batch_size`, `num_workers`, `seed` | Optimizer / data loading |
| `d`, `layers`, `heads`, `dropout` | GNN architecture |
| `sinkhorn_w`, `pair_w`, `nce_w`, `dust_w`, `dust_pos_w`, `nce_tau`, `sinkhorn_iters` | Loss weights |
| `fu_jitter`, `p_drop_fu`, `p_drop_bl`, `k_intra` | Augmentation + graph kNN (ignored when `intra=complete`) |
| `drop_dp` | Zero the 5 registered `dp/dist` channels in `cross_attr`. Retrain required. |
| `intra` | `"knn"` or `"complete"` (deployed: complete) |
| `type_mask` | Restrict intra edges to the same `lesion_type` |
| `ema_decay`, `ema_start_step`, `val_score_ema_beta` | Weight EMA + smoothed val score for early stop |
| `dust_tau` | Decode dustbin threshold. Inference default **0.125**; train JSON may differ |
| `n_folds`, `cv_seed` | Patient-level k-fold CV |

## What the CLI takes

- **Config-driven CLIs** (`lesionglue_train`, `lesionglue_cv`, `lesionglue_report`): pass `--config lesionglue/configs/base.json`. Training writes a copy to `{out}/config.json`.
- **CLI-only overrides:** paths (`--root`, `--cache`, `--out`), W&B flags, fold index (`--fold`), early-stop disable, eval/report device and batch settings.
- **`lesionglue_track` decode flags:** `--decode`, `--thresh`, `--sinkhorn-tau`, `--sinkhorn-iters` (see [track.md](../steps/track.md)).
- Copy and edit `lesionglue/configs/base.json` for experiments; unknown keys raise on load (`unknown config keys:`).
