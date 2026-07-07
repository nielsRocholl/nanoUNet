# Handoff: run uniGradICON registration on the GPU cluster

**For the next agent (has GPU-cluster access).** A `unigradicon` backend was added to
`nanounet_register_longi` alongside the classical `elastix` backend. It produces **byte-compatible
output** (identical `inputsTrBL/`, `targetsTrBL/`, warped-BL click JSON, QC) — only the registration
engine changes. Your job: run it end-to-end on real data and validate the clicks. **It has not yet
been run end-to-end anywhere** (the dev container is CPU-only and its egress blocked the weight
download), so you are the first real run.

Branch: `claude/nanonet-unigradicon-registration-0busio`

## What changed (code is in place, deps resolve)

| File | Change |
|------|--------|
| `nanounet/register/unigradicon.py` | New backend: CT→[0,1] preprocess, `register_pair`, warp orig img/seg/ids via `resample_to`; native IO = refinement |
| `nanounet/register/elastix.py` | Added `warp_pair` (packages the existing elastix register+resample) |
| `nanounet/register/warp_case.py` | `if backend=="elastix" … else …`; local elastix `refine` runs **only** for elastix |
| `nanounet/cli/register_longi.py` | `--backend`, `--io-iterations`; startup validation (package + weights); config table |
| `pyproject.toml` | `unigradicon>=1.0.4` (pulls `icon_registration`, `itk==5.4.6` — coexists with `itk-elastix`, no pin) |

## First: get the weights

uniGradICON downloads weights on first use to `$NANOUNET_UNIGRADICON_WEIGHTS` (default
`~/.cache/nanounet/unigradicon/Step_2_final.trch`). If the cluster **can** reach github, the CLI
fetches them automatically at startup. If cluster nodes have **no internet** (common), pre-stage once
from a login node with egress:

```bash
export NANOUNET_UNIGRADICON_WEIGHTS=/shared/weights/unigradicon/Step_2_final.trch
mkdir -p "$(dirname "$NANOUNET_UNIGRADICON_WEIGHTS")"
curl -L -o "$NANOUNET_UNIGRADICON_WEIGHTS" \
  https://github.com/uncbiag/uniGradICON/releases/download/unigradicon_weights/Step_2_final.trch
```
Then export that same env var in every job. Missing/blocked weights now fail at startup with this
exact fix printed (not a mid-batch traceback).

## Run it

Single pair (start here — eyeball QC before scaling):
```bash
uv run nanounet_register_longi \
  --data-root /path/to/raw --out /path/to/regout_ug \
  --pid <PID> --idx 00 --qc \
  --backend unigradicon --io-iterations 50
```

Batch (all pairs):
```bash
uv run nanounet_register_longi \
  --data-root /path/to/raw --out /path/to/regout_ug --all --qc \
  --backend unigradicon --io-iterations 50
```

- `--io-iterations 50` = uniGradICON's native **instance optimization** (default; the "best clicks"
  step). It is 50 backprop passes through a 3D UNet at 175³ per pair → **wants a GPU**. `0` disables
  IO (fast, lower accuracy). Try 100 if you want to trade time for accuracy.
- The model auto-uses CUDA when visible (`torch.cuda.is_available()`); no device flag. Run **one case
  per GPU process**. To parallelize, mirror `scripts/slurm_nanounet_register_longi.sh` (per-case
  subprocess with a job throttle) but add `--backend unigradicon --io-iterations 50`; the `--threads`
  flag there is elastix-only and harmless to leave/drop for unigradicon.
- `--no-refine`/`--no-body-mask` are **elastix-only** and ignored here (IO is the refinement).

## Validate (this is the point of the run)

1. Exit 0, and per case it wrote: `inputsTrBL/<case>.nii.gz` + `.json`, `targetsTrBL/<case>.nii.gz`,
   copied `inputsTrFU/…`, and `qc/<case>_qc.png`.
2. Open a few QC PNGs: the yellow BL clicks should land **inside** the FU lesions and the warped-BL
   mask contour should track FU anatomy.
3. Compare against elastix on the same cases — run once with `--backend elastix` into a sibling `--out`
   and diff structure + click displacement:
   ```bash
   uv run nanounet_register_longi --data-root /path/to/raw --out /path/to/regout_el --all --qc
   # same file set, same JSON schema; clicks differ by method — compare which lands better in QC
   ```
4. Downstream is unchanged: feed `regout_ug` into `nanounet_longi_build` → `nanounet_longi_clicks`
   exactly as with elastix (see `docs/steps/longi.md`).

## Notes / gotchas

- **CPU segfault seen in the dev sandbox is NOT relevant to you.** The elastix CLI segfaulted there
  only because a sandbox syscall restriction trips itk-elastix's threader when `number_of_threads` is
  set explicitly; it reproduces on pre-diff code and won't occur on a normal GPU node.
- Deps verified to co-resolve (`uv sync` → `itk==5.4.6` for both itk-elastix and icon_registration).
- uniGradICON expects `itk.Image` (handled internally) and CT in HU; our `preprocess(img,"ct")` clips
  HU to [-1000,1000] and scales to [0,1] for the network only — the **original** HU/labels are what
  get warped and written, so output HU is preserved.
- If a case fails, it's skipped with a reason and the batch continues (exit code nonzero if any
  skipped), same as elastix.

## If you improve things

Report per-case timing (the results table already prints it) and, if you tune `--io-iterations`,
note the accuracy/time trade-off. Keep any code edits within nanochat-style
(`.claude/skills/nanochat-style`): <200 LOC/file, errors that teach, no fallbacks.
