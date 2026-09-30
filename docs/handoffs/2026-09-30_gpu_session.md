Date: 2026-09-30
Status: open. Paste everything below the line into a fresh agent session on the GPU cluster.

---

You are working on the GPU cluster in my repo `nanoUNet` (GitHub `git@github.com:nielsRocholl/nanoUNet.git`).
You have no prior context. The checkout on this machine is OLD. Everything you need is on the branch
`monorepo`. Work in small, verified steps and report in dense tables.

## 0. Ground rules

- Never touch `main`. Never force-push. Commit only on `monorepo`. Push only `git push origin monorepo`.
- Never delete or overwrite anything under `/data/oncology/experiments/universal-lesion-segmentation`
  (mounted as `/nnunet_data` in containers) except new output dirs you create under
  `/nnunet_data/lesion_tracking/runs/gpu_check_2026-09-30/` and
  `/nnunet_data/NanoUNet_results/gpu_check_2026-09-30/`.
- Do not cancel or disturb other running Slurm jobs (`squeue -u $USER` first; note what is running).
- Commits end with `Co-Authored-By: Claude <noreply@anthropic.com>`. One concern per commit.
- If something is ambiguous or a gate fails, stop and report. Don't paper over it.

## 1. Get the right code

1. Find the checkout. Expected: `/home/nielsrocholl/projects/git_projects/nanoUNet`. If it is elsewhere, `find ~ -maxdepth 4 -name .git -path "*nanoUNet*"`.
2. `git status` and `git stash list`. If the tree has local changes, do NOT discard them:
   `git switch -c backup-cluster-$(date +%Y%m%d) && git add -A && git commit -m "backup: cluster local changes"`, then continue.
3. `git fetch origin && git switch monorepo` (or `git switch -c monorepo origin/monorepo`), then `git pull --ff-only`.
   Confirm `git log --oneline -1` matches `origin/monorepo`.
4. Read, in order: `README.md` (project map), `docs/dev-notes/monorepo_plan.md` (what changed, gates, what is open),
   `.claude/skills/nanochat-style/SKILL.md` (the coding standard; non-negotiable), and
   `.claude/skills/nanochat-style/references/gpu.md` (measurement protocol G4).

What changed since the old checkout (so you are not surprised):
- The repo is now a monorepo with one package per project: `nanounet/` (segmentation), `lesionglue/`
  (lesion matching, formerly the separate `lesion-tracking` repo, package `tracking`), `segtrack/` (pipeline),
  `core/` (shared terminal UI). Each project has its own `README.md`, `docs/`, `configs/`, `scripts/`.
- nanounet configs moved: `configs/*.json` → `nanounet/configs/*.json`. Old job scripts that pass
  `--config configs/...` must now pass `--config nanounet/configs/...`.
- Console scripts: `nanounet_*`, `lesionglue_*` (13: split, preprocess, train, cv, oof, pool, eval, report,
  predict, track, audit, qc, baseline_distance), `segtrack_run` (was `nanounet_segtrack`).
  `lesion_track*` → `lesionglue_*`. On-disk/ckpt names are unchanged (R18): old checkpoints load.
- All human output goes to stderr; every command prints a header, a config table and a `next:` line.

## 2. Environment

Work inside the project container (same as the jobs), not on the login node. Container flags are in
`lesionglue/scripts/lesion-round9-cv.sh` and `nanounet/scripts/slurm_final_900_h200.sh`
(image `dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest`, mount
`/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data` plus the repo path).
Use `srun --pty` with those flags for interactive checks, `sbatch` for anything > 15 min.
Inside the container: `cd <repo> && pip install -e ".[lesionglue]"`, then:

```bash
python .claude/skills/nanochat-style/scripts/check.py      # expect: 0 error(s)
for c in split preprocess train cv oof pool eval report predict track audit qc baseline_distance; do lesionglue_$c --help >/dev/null || echo "FAIL $c"; done
nanounet_train --help >/dev/null && segtrack_run --help >/dev/null && echo ok
nvidia-smi
```

nanounet env (from the slurm scripts): `NANOUNET_RAW=/nnunet_data/NanoUNet_raw`,
`NANOUNET_RESULTS=/nnunet_data/NanoUNet_results`, `NANOUNET_PREPROCESSED` = the preprocessed dir the
slurm script stages (read it from `nanounet/scripts/slurm_final_900_h200.sh`; use the NFS copy if no local
staging), `NANOUNET_TMPDIR` on local disk.

## 3. Task A: real-data smoke of the refactor (gates first, before any change)

The refactor was verified locally with goldens, but these paths need data or a GPU. Run each and record
exit code, wall time, and the last ~15 stderr lines. Anything that errors is a finding. Don't fix it in this step.

| # | command (adapt paths only) | expect |
|---|---|---|
| A1 | `lesionglue_eval --split test` (deployed ckpt, EMA, default decode) | test match reproduces the deployed number: `0.9701` (57 graphs), see `lesionglue/README.md`. This is the key gate. |
| A2 | `lesionglue_eval --split val --dust-tau 0.10 --dust-tau 0.125 --dust-tau 0.20 --out /nnunet_data/lesion_tracking/runs/gpu_check_2026-09-30/tau.json` | JSON has `rows` and `selected.dust_tau` (round9.sh now depends on this) |
| A3 | `lesionglue_train --config lesionglue/configs/base.json --fold 0 --max-steps 200 --no-early-stop --out /nnunet_data/lesion_tracking/runs/gpu_check_2026-09-30/train_f0` | runs validation under a real Trainer: `val_match_score` logged; validation hooks now live in `lesionglue/train/val_score.py`, bound onto `MatcherModule` |
| A4 | `lesionglue_oof` + `lesionglue_pool` on A3's output (use the `next:` lines the commands print) | `val_per_patient.json`, `pool_summary.json` written |
| A5 | `lesionglue_track --split test --root /nnunet_data/Longitudinal-CT --out /nnunet_data/lesion_tracking/runs/gpu_check_2026-09-30/track` | one CSV per patient, columns `bl_lesion_id,fu_lesion_id,pair_prob,decode,track_id` |
| A6 | `nanounet_train` 2-epoch smoke on the same dataset/plans as `slurm_final_900_h200.sh`, small `--iters-per-epoch`, `--out` under `gpu_check_2026-09-30` | header, config table, `epoch_wall_time_sec` logged, `next:` line |
| A7 | `segtrack_run` on one BL/FU case (see `segtrack/README.md` for the literal command) | linked masks written |

## 4. Task B: G2 host-sync findings (the only open code items; they need a GPU number)

The checker flags CPU-GPU syncs in hot paths (`python .claude/skills/nanochat-style/scripts/check.py | grep G2`):

1. `lesionglue/model/matcher.py` `_dust_graph` (~lines 103-106): `torch.split(..., (nb * nf).tolist())` and
   `nb.tolist()` / `nf.tolist()`. These run on every training step.
2. `nanounet/model/loss/cc_dice_ce.py` `_cc_term` (~lines 68, 86, 109): `.cpu().numpy()` for connected
   components plus `.item()` checks. Only active with `nanounet_train --loss cc_dc_ce` (production uses `dc_ce`).

For each one, follow G4 in `references/gpu.md`: measure before touching code.
- lesionglue has no `epoch_wall_time_sec`. Measure per-step time as
  `(t(--max-steps 600) - t(--max-steps 100)) / 500` with `lesionglue_train --fold 0 --no-early-stop`, 3 repeats
  per arm, same node/GPU, report the median. Also `nvidia-smi dmon -s u -d 5` GPU util.
- nanounet: `--epochs 4`, discard epoch 0, median of e1-e3 `epoch_wall_time_sec`, fixed everything else.

Decide per finding, then report the table:
- If removing the sync is possible without changing results (e.g. split sizes computed on CPU before the
  batch moves to GPU, or from data already on host), implement it as its own commit, prove numerics are
  unchanged (same seed: identical loss curve for 50 steps, or bit-identical forward on a fixed batch), and put the
  before/after table in the commit message. Reject it if the gain is < ~2% (noise). Say so and waive instead.
- If the sync is inherent (CPU connected components via scipy in `cc_dc_ce`), measure what the loss costs
  vs `dc_ce`, then waive it inline with the number:
  `# nanochat-style: allow G2 (CPU CC labelling is inherent; cc_dc_ce costs +X% epoch time on H200, measured 2026-..)`
  and make the `--loss` help text state the measured cost.

## 5. Finish

- `python .claude/skills/nanochat-style/scripts/check.py` → 0 errors; G2 warns gone or waived with numbers.
- Update `docs/dev-notes/monorepo_plan.md`: add the Task A results table and Task B before/after tables; move G2 out of "Still open".
- Commit (one concern per commit), `git push origin monorepo`.
- Report to me: table of A1-A7 (status, key number, time), G2 tables, commits, anything that failed and why.
