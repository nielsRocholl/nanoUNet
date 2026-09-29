# SegTrack inference + honest split

Implementer spec. No guesswork. nanochat-style (R1–R16, U1–U7, E1–E5). Packages stay separate: `tracking` is importable; nanoUNet calls it. Do not merge repos.

Bible: `/lesion-tracking/.cursor/skills/nanochat-style/SKILL.md`.

---

## Persistence (hard rule — container is ephemeral)

Code lives on a compute-node Docker container. **Session stop = disk wipe for the repo.** The only durable volume is `/nnunet_data`. GitHub is the durable copy of source.

| What | Where it must live |
|------|-------------------|
| Source (`/lesion-tracking`, `/nanoUNet`) | GitHub. Commit + **push** after every working slice. Unpushed work is gone if the session dies. |
| Checkpoints, graph cache, train logs | `/nnunet_data/lesion_tracking/` (already the plan’s `--out` / `CACHE_ROOT`) |
| `configs/split.json` | **repo** (committed). Not only on `nnunet_data`. |

Rules for the implementer:

1. After each item in §11 (and after any other coherent edit), `git add` the intended files, commit, **`git push`**. Do this **before** kicking a long GPU job so a killed session does not strand unpublished train/CLI code next to a ckpt nobody can rerun.
2. Two repos. Push **both** if both changed: `/lesion-tracking` and `/nanoUNet`.
3. Never `--force` to `main`/`master`. Never `--no-verify`. Never `git config`.
4. Do not leave the only copy of a new file in `/tmp` or the container home.
5. Probe scripts (§7) are deleted after the number is recorded — record the number in the commit message of the push that removes them, or in README, not in a local scrap.
6. If push needs credentials/network and fails: **stop and say so**. Do not keep stacking unpushed commits.

---

## 0. Goal / non-goals

**Goal.** Tracking is easy, GPU-fast, and honest for the nanoUNet holdout (`test_patients.csv`, 60 patients). One Python API + one CLI. Seg stays in nanoUNet; tracking consumes instance masks.

**Non-goals.** Matcher architecture. R11 geometry. pyradiomics. Type classifier. Rewriting `technical.md`. Dash QC. Dual configs. Overwriting `/nnunet_data/Longitudinal-CT/data_split.json`.

**Facts (do not rediscover).**
- Features: Yerebakan L0 1372 + 14 mask stats + type = 1387. Not pyradiomics.
- Dataset root on disk: `/nnunet_data/Longitudinal-CT/` (code default path does not exist).
- Official JSON split: 240 / 30 / 30. `test_patients.csv` = val ∪ test = 60.
- CV today pools train+val (270) → trains on 22–29 of the 60. Invalid for E2E.
- Existing `r12_selector` / `graph-based-tracking/models/best.ckpt` are not this split.
- `predict_masks.py` is CLI-only, CPU, always Hungarian.
- nanoUNet predict writes **binary** FG. Tracking needs `voxel = lesion_id`.
- Dataset999 ckpt is single-stream ResEnc, 2-class logits. Loadable.
- Canonical hyperparams file on disk: `configs/r9_base.json`. `configs/base.json` does not exist. Do not create a second file.

---

## 1. Split (do this first)

Do **not** edit `Longitudinal-CT/data_split.json`.

Write `/lesion-tracking/configs/split.json` via a function, then commit the file.

```python
# tracking/data/splits.py — replace pool_patient_ids; add these fns

HOLDOUT_CSV = Path("/nnunet_data/Longitudinal-CT/test_patients.csv")
OFFICIAL_SPLIT = Path("/nnunet_data/Longitudinal-CT/data_split.json")
SPLIT_PATH = Path("configs/split.json")  # repo-relative default

def load_holdout(csv_path: Path) -> list[str]:
    # CSV header `patient`. Fail with 3-question error if missing/empty.

def build_split(dataset_root: Path, holdout_csv: Path, n_folds: int = 5, seed: int = 0, val_fold: int = 0) -> dict:
    official = load_split_json(Path(dataset_root) / "data_split.json")
    holdout = set(load_holdout(holdout_csv))
    train240 = [str(x) for x in official["train"]]
    leak = set(train240) & holdout
    if leak:
        raise SystemExit(f"{len(leak)} holdout ids in official train: {sorted(leak)[:8]}...")
    extra = (set(map(str, official["val"])) | set(map(str, official["test"]))) - holdout
    if extra:
        raise SystemExit(f"official val/test id not in holdout csv: {sorted(extra)}")
    missing = holdout - set(map(str, official["val"])) - set(map(str, official["test"]))
    if missing:
        raise SystemExit(f"holdout id not in official val∪test: {sorted(missing)}")
    fm = fold_map(train240, n_folds, seed)
    val = sorted(p for p, f in fm.items() if f == val_fold)
    train = sorted(p for p, f in fm.items() if f != val_fold)
    test = sorted(holdout)
    assert not (set(train) | set(val)) & set(test)
    return {"train": train, "val": val, "test": test, "n_folds": n_folds, "seed": seed, "val_fold": val_fold}
```

`pool_patient_ids` **must** become: ids in `configs/split.json` train ∪ val (the 240). Never official val.

`LesionDataset.process` reads `configs/split.json` (pass path; default repo `configs/split.json`). 3-question error if missing:

```
No tracking split at {path}.
Expected output of: python3 tracking/cli/split.py
Fix: python3 tracking/cli/split.py --root /nnunet_data/Longitudinal-CT --holdout /nnunet_data/Longitudinal-CT/test_patients.csv --out configs/split.json
```

Bump `CACHE_TAG` in `tracking/data/features.py` from `"v5_l0"` to `"v6_h60"` so old leaked caches cannot load.

`DATASET_ROOT` in `tracking/common.py` → `Path("/nnunet_data/Longitudinal-CT")`.
`CACHE_ROOT` → `Path("/nnunet_data/lesion_tracking/cache")`.

Tiny CLI `tracking/cli/split.py` (argparse + `dump_json`, ~40 LOC): writes the file, prints a 3-row table (split, n, source). Then preprocess:

```
python3 tracking/cli/preprocess.py --split all --jobs 16
```

Delete any leftover `{split}_v5_l0.pt` only if they sit in the same cache dir and would confuse operators — do not write a cleanup script; one `cprint` in preprocess if a `v5_l0` file is found: tell the user to ignore/delete.

Startup assert in `MatcherDataModule.prepare_data`: after loading split, `set(train)|set(val)` ∩ holdout == ∅. Crash before first step.

---

## 2. Rich surface (`tracking/common.py`)

`print0` stays as rank-0 gate. Add, copied in spirit from `nanounet/common.py` (do not import nanounet):

```python
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

_CONSOLE = Console(stderr=True)

def cprint(msg, **kw): ...
def nano_header(title: str, color: str = "cyan") -> None: ...
def config_table(rows: list[tuple[str, object, str]], title: str = "config") -> None: ...
```

`print0 = cprint`. No bare `print` outside `common.py`. Train CLI: drop `TQDMProgressBar`, use Lightning's default quiet + rich header/summary.

3-question errors at every user boundary (missing ckpt, missing NIfTI, split mismatch, bad `--decode`).

---

## 3. Decode (product UX)

Three methods. Names are the CLI values. Write this table into the CLI help **and** the interactive panel — same words.

| choice | what it does | merges (many BL → one FU) | splits (one BL → many FU) |
|--------|----------------|---------------------------|---------------------------|
| `dense` | keep every BL–FU pair with `sigmoid(pair) ≥ thresh` | kept | kept |
| `sinkhorn` | SuperGlue Sinkhorn, each BL picks ≤1 FU (or none / dustbin) | kept | dropped |
| `hungarian` | Sinkhorn then Hungarian 1-to-1 | dropped | dropped |

`dense` is the model’s native output (pair logits + dustbins). `sinkhorn` / `hungarian` are optional post-processes. Never imply Hungarian is “the” tracking result.

### 3.1 Code

`tracking/decode.py` — add `decode_dense`. Keep the two existing fns.

```python
DECODE_CHOICES = ("dense", "sinkhorn", "hungarian")

def decode_dense(pair_log: torch.Tensor, n_bl: int, n_fu: int, thresh: float = 0.5) -> np.ndarray:
    """(n_bl,) object-unfriendly: return int64 array shape (E, 2) of (i, j) with sigmoid >= thresh."""
    p = torch.sigmoid(pair_log).reshape(n_bl, n_fu)
    ii, jj = (p >= thresh).nonzero(as_tuple=True)
    return torch.stack([ii, jj], dim=1).cpu().numpy().astype(np.int64)

def decode_pairs(method: str, pair_log, dust_bl, dust_fu, n_bl, n_fu, *, thresh, sinkhorn_iters, sinkhorn_tau) -> np.ndarray:
    assert method in DECODE_CHOICES
    if method == "dense":
        return decode_dense(pair_log, n_bl, n_fu, thresh)
    fn = decode_sinkhorn if method == "sinkhorn" else decode_sinkhorn_hungarian
    asg = fn(pair_log, dust_bl, dust_fu, n_bl, n_fu, iters=sinkhorn_iters, tau=sinkhorn_tau)
    # asg[i] = j or -1  →  (E, 2) live pairs
    live = [(i, int(j)) for i, j in enumerate(asg) if int(j) >= 0]
    return np.asarray(live, dtype=np.int64).reshape(-1, 2)
```

Return type of `track()` always includes the dense `pair` logits (N×M) and dustbins, **plus** `pairs` from `decode_pairs`. Caller can re-decode.

### 3.2 CLI resolve (write this exactly)

`--decode {dense,sinkhorn,hungarian}` optional.

```python
def resolve_decode(cli_value: str | None) -> str:
    if cli_value is not None:
        return cli_value
    import sys
    from rich.prompt import Prompt
    if not sys.stdin.isatty():
        raise SystemExit(
            "No --decode given and stdin is not a TTY.\n"
            "Expected one of: dense, sinkhorn, hungarian.\n"
            "Fix: lesion_track ... --decode dense"
        )
    t = Table(title="How should matches be decoded?", box=None, padding=(0, 2))
    t.add_column("choice", style="cyan")
    t.add_column("keeps")
    t.add_column("drops")
    t.add_row("dense", "every pair above threshold; one lesion can match many", "nothing (native model output)")
    t.add_row("sinkhorn", "each baseline → at most one follow-up; merges stay", "splits (one BL → many FU)")
    t.add_row("hungarian", "strict 1-to-1 list", "merges and splits")
    cprint(t)
    cprint("[dim]dense = what the network outputs. hungarian / sinkhorn = optional post-process.[/dim]")
    return Prompt.ask("decode", choices=["dense", "sinkhorn", "hungarian"])
```

No silent default. TTY → must pick. Non-TTY → must pass `--decode`. Python API: `decode: str` is a **required** keyword. No Prompt inside `track()`.

Argparse help string (verbatim):

```
how to turn pair logits into matches: dense (keep all pairs above --thresh; merges and splits stay), sinkhorn (each baseline picks at most one follow-up; merges stay, splits drop), hungarian (strict 1-to-1; merges and splits drop). Omit to choose interactively.
```

---

## 4. Inference API

New file `tracking/infer.py` (<200 LOC). This is the only function nanoUNet imports.

```python
@dataclass
class TrackResult:
    bl_ids: np.ndarray          # (N,)
    fu_ids: np.ndarray          # (M,)
    pair: np.ndarray            # (N, M) logits
    pair_prob: np.ndarray       # (N, M) sigmoid
    dust_bl: np.ndarray         # (N,)
    dust_fu: np.ndarray         # (M,)
    pairs: np.ndarray           # (E, 2) decoded (i, j) into bl_ids/fu_ids index space
    decode: str

def track(
    bl_img: Path, bl_mask: Path, fu_img: Path, fu_mask: Path,
    propagated,  # Path to CSV or (lesion_id, z, y, x) array — if array, write a tiny helper that builds the same graph path
    ckpt: Path,
    *,
    decode: str,
    device: str = "cuda",
    default_lesion_type: str | None = "unclear",
    k_intra: int = 8,
    thresh: float = 0.5,
    sinkhorn_iters: int = 20,
    sinkhorn_tau: float = 0.2,
    use_ema: bool = True,
) -> TrackResult:
```

Body, in order:
1. `assert decode in DECODE_CHOICES`
2. `dev = eval_device(device)` (already in `common.py`)
3. `data = build_mask_graph(...)`  # existing
4. `mod = MatcherModule.load_from_checkpoint(str(ckpt), map_location=dev); mod.to(dev).eval()`
5. `bat = Batch.from_data_list([data.to(dev)])`
6. `with torch.no_grad(): outp = mod.predict_batch(bat, use_ema=use_ema)`
7. reshape `outp.pair` to (N, M); decode; return `TrackResult`

If `propagated` is a Path, existing CSV loader. If ndarray, require columns aligned with `bl` mask labels — simplest: only Path in v1, ndarray later if it costs lines. **v1 = Path CSV only.**

`tracking/__init__.py` exports `track`, `TrackResult`, `DECODE_CHOICES`.

### CLI `tracking/cli/track.py`

`main()` = argparse + `resolve_decode` + `config_table` + `track()` + write CSV + summary. No business logic.

Entry point in `pyproject.toml`:

```
[project.scripts]
lesion_track = "tracking.cli.track:main"
lesion_track_split = "tracking.cli.split:main"
lesion_track_preprocess = "tracking.cli.preprocess:main"
lesion_track_train = "tracking.cli.train:main"
lesion_track_eval = "tracking.cli.eval:main"
```

Each of those CLIs needs a `main()` (R13). Add `def main():` that is argparse + call; keep files under 200.

CLI args (deploy):

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bl-img` `--bl-mask` `--fu-img` `--fu-mask` | path | required | NIfTI |
| `--propagated` | path | required | CSV `lesion_id,z,y,x` (+ optional `lesion_type`) |
| `--ckpt` | path | required | Lightning ckpt |
| `--out` | path | required | matches CSV |
| `--decode` | choice | unset | see §3 |
| `--thresh` | float | 0.5 | dense pair cutoff |
| `--device` | `cuda\|cpu\|mps` | `cuda` | |
| `--k-intra` | int | 8 | |
| `--sinkhorn-iters` | int | 20 | |
| `--sinkhorn-tau` | float | 0.2 | |
| `--default-lesion-type` | str | `unclear` | |
| `--no-ema` | flag | off | |
| `--pairs-out` | path | `""` | optional full N×M dump |

Output CSV columns: `bl_lesion_id, fu_lesion_id, pair_prob, decode`. One row per decoded pair. Empty decode → file with header only.

Header: `nano_header("lesion_track")`. Summary: n_bl, n_fu, n_pairs, decode, out path.

Delete `tracking/cli/predict_masks.py` after `track.py` works. Leave `predict.py` as **cached-graph benchmark only**; point its docstring at `lesion_track` for deployment. Do not teach two deploy CLIs.

Train/eval/preprocess: add `main()`, rich header, 3-question missing-cache errors. Fix `--config` default help to `configs/r9_base.json`.

---

## 5. Instance bridge

New `tracking/data/instances.py`. nanoUNet writes binary FG. Tracking needs instance ids.

Clicks JSON (nanoUNet): `{"points": [{"name": "<int lesion_id>", "point": [x, y, z]}, ...]}` — **x,y,z scanner voxels**. Convert to z,y,x with the same round-clip as `nanounet/score.py` lines 109–114.

```python
def binary_to_instances(pred: np.ndarray, clicks_zyx: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """pred is bool/0-1, same grid as clicks. Return int32 mask, voxel = lesion_id."""
    import cc3d
    out = np.zeros(pred.shape, dtype=np.int32)
    lab = cc3d.connected_components((pred > 0).astype(np.uint8), connectivity=18)
    claimed: dict[int, int] = {}  # cc_id -> lesion_id
    conflicts: list[tuple[int, int, int]] = []  # (lesion_id, other_id, cc)
    for lid, (z, y, x) in clicks_zyx.items():
        z = min(max(int(z), 0), pred.shape[0] - 1)
        y = min(max(int(y), 0), pred.shape[1] - 1)
        x = min(max(int(x), 0), pred.shape[2] - 1)
        cc = int(lab[z, y, x])
        if cc == 0:
            continue  # click missed FG; do not steal
        if cc in claimed and claimed[cc] != lid:
            conflicts.append((lid, claimed[cc], cc))
            continue
        claimed[cc] = lid
        out[lab == cc] = lid
    return out  # caller may cprint conflicts; do not silence
```

Also `instances_from_nifti(pred_path, clicks_json_path, out_path)` using nibabel (tracking already depends on it; do not add SimpleITK to tracking). Copy affine from pred.

Add dep `connected-components-3d>=3.12` to `pyproject.toml`.

No GT in this path. Do not copy `score_case` overlap-with-GT fallback — that is scoring, not serving.

CLI flag on `lesion_track`: optional `--bl-clicks` / `--fu-clicks` JSON. If set, those masks are treated as binary and converted in-memory before `build_mask_graph`. If unset, masks are already instance.

---

## 6. Train on the new split

After cache rebuild:

```
lesion_track_train --config configs/r9_base.json \
  --out /nnunet_data/lesion_tracking/runs/h60_r9 \
  --wandb --wandb-project lesion-tracking --wandb-run-name h60_r9
```

This is the **final** model: train = split.json train, val = split.json val (a fold of the 240). Early-stop on that val. **Do not** look at the 60 until eval.

Optional same-day: `lesion_track` CV over the 240 (`--fold`) if train of the final run is kicked off first. Do not block inference work on CV.

`num_workers`: this box is H200 + ample CPU. Set config `num_workers` to 8 for the run (edit `r9_base.json` `num_workers` 2 → 8). That is a real default for this machine, not a fallback.

Eval once, after train:

```
lesion_track_eval --ckpt /nnunet_data/lesion_tracking/runs/h60_r9/best.ckpt --split test
```

That is the **oracle-mask** tracking number on the 60.

---

## 7. Encoding probe (measure, then stop)

Throwaway script. Write, run, delete (R16). Do not leave `tests/` or `scripts/profile_*.py`.

On **one** cached val graph + one raw case:

1. Time `descriptor_l0` + `mask_stats` for all lesions (CPU).
2. Time `build_mask_graph` wall.
3. Time `track()` GPU forward only (exclude I/O).
4. If easy (<40 LOC hook): `net.encoder(patch)` GAP over pred FG in that patch, once, no TTA. Record dim + ms. Do **not** retrain a matcher on deep feats in this session unless (4) is both faster than L0 **and** you have leftover GPU after §6.

Print a 4-row table. Keep L0 as default. Report numbers in the session recap.

Do not change `FEAT_DIM` / `pack_node` unless we explicitly decide to after the table.

---

## 8. nanoUNet hook (thin)

New `nanounet/cli/segtrack.py` + `nanounet_segtrack` entry in `/nanoUNet/pyproject.toml`.

Does **not** reimplement predict. Sequence:

1. User already has binary preds (or we call existing library fns). Preferred v1: **paths in, no nested trainer**.
2. `instances_from_nifti` on BL and FU.
3. `from tracking.infer import track`
4. Write tracking CSV next to preds.

```
nanounet_segtrack \
  --bl-img ... --bl-pred ... --bl-clicks ... \
  --fu-img ... --fu-pred ... --fu-clicks ... \
  --propagated ... \
  --track-ckpt /nnunet_data/lesion_tracking/runs/h60_r9/best.ckpt \
  --decode dense --out ...
```

`--decode` same resolve as lesion_track (import `resolve_decode` from tracking, or duplicate the 15-line fn — **import**).

If `tracking` is not installed: 3-question error `pip install -e /lesion-tracking`.

Do not add optional-deps circus. Do not subprocess `lesion_track`.

Docs: `docs/steps/predict.md` — 10-line “then track” pointing at this command. Argument table required (D3). File must stay <200 LOC; if tight, new `docs/steps/track.md` linked from index.

After code change in nanoUNet: `graphify update .`

---

## 9. Eval (oracle first)

Oracle-mask: `lesion_track_eval --split test` on GT instance masks of the 60. That is today’s number.

E2E (seg then track) only if §4–§6+§8 exist: Dataset999 ckpt, `--patients-csv test_patients.csv`, instance bridge, then `track`. Identity scoring = pred instance ↔ GT instance via IoU>0.1 (same `IOU_HIT` as `nanounet.score`), then pair vs meta topology. If this does not fit, stop after oracle and say so. Do not invent a metrics framework.

---

## 10. File list

| Path | Action |
|------|--------|
| `tracking/common.py` | `DATASET_ROOT`/`CACHE_ROOT`; add `cprint`/`nano_header`/`config_table` |
| `tracking/data/features.py` | `CACHE_TAG = "v6_h60"` |
| `tracking/data/splits.py` | `build_split`, `load_holdout`; `pool_patient_ids` from `configs/split.json` |
| `tracking/data/dataset.py` | read `configs/split.json`; 3-question error |
| `tracking/data/instances.py` | **new** binary→instance |
| `tracking/decode.py` | `decode_dense`, `decode_pairs`, `DECODE_CHOICES` |
| `tracking/infer.py` | **new** `track` / `TrackResult` |
| `tracking/cli/split.py` | **new** |
| `tracking/cli/track.py` | **new** deploy CLI + `resolve_decode` |
| `tracking/cli/predict_masks.py` | **delete** after track.py works |
| `tracking/cli/{preprocess,train,eval,cv}.py` | `main()`; rich; config path `r9_base.json` |
| `tracking/__init__.py` | export `track`, `TrackResult` |
| `tracking/train/datamodule.py` | holdout disjointness assert |
| `configs/split.json` | **new**, generated |
| `pyproject.toml` | scripts + `connected-components-3d` |
| `README.md` | replace `base.json`, `PYTHONPATH=.`, `predict_masks.py`; argument tables |
| `nanounet/cli/segtrack.py` | **new** |
| `nanoUNet/pyproject.toml` | `nanounet_segtrack` |
| `nanoUNet/docs/steps/track.md` | **new** if predict.md is full |

Do not touch `.cursor/plans/` except this file. Do not reintroduce FeatConfig / MAE / TTA.

---

## 11. Order

**Push after every step.** Step 2 starts GPU: push step 1 first.

1. Split fn + `configs/split.json` + `CACHE_TAG` + `DATASET_ROOT`. Assert 240 ∩ 60 = ∅. **Push.**
2. Kick preprocess (`--jobs 16`) then train (§6) on GPU. Do not wait.
3. `common.py` rich + `infer.py` + `decode` + `cli/track.py` + pyproject scripts. Delete `predict_masks.py`. **Push.**
4. `instances.py` + `--bl-clicks`/`--fu-clicks`. **Push.**
5. Encoding table (§7). Delete the probe script. **Push.**
6. `nanounet_segtrack` + doc. **Push nanoUNet.**
7. Oracle eval on the 60 when ckpt exists. Numbers + any eval CLI fixes: **push.**

---

## 12. Review checklist (every PR-sized change)

- [ ] Every touched file <200 LOC, module docstring, no banner comments
- [ ] No new ABC / factory / `utils/`
- [ ] Output through `cprint` / `nano_header` / Table
- [ ] Boundary errors: what / expected / fix command
- [ ] Train/val ∩ `test_patients.csv` is empty (assert fired at startup)
- [ ] `--decode` omitted → interactive table; non-TTY → hard error; API requires `decode=`
- [ ] Default deploy path does not Hungarian-force
- [ ] Data-path change → report `build_mask_graph` ms and `track()` ms
- [ ] CLI/path change → README or `docs/steps/track.md` argument table in the same change
- [ ] This slice is committed **and pushed** to GitHub (both repos if both changed). Container-only copies do not count.
