# Seg-track E2E — living plan

Implementer spec. No guesswork. nanochat-style bible: `/lesion-tracking/.cursor/skills/nanochat-style/SKILL.md` (same as nanoUNet).

**Status:** Part 1 (output) locked. Part 2 (CLI + code) drafted — wait for sign-off before coding.

Packages stay separate: seg in nanoUNet, tracking in `lesion-tracking`. Do not merge repos.

---

## Part 1 — Output

What one case writes. CSV stays. Masks are the product view: same integer on both scans = same lesion.

### 1.1 Files (one case)

`--out` is a **directory** (CLI flags locked in Part 2). Always write all three:

| File | Dtype / format | Grid | Meaning |
|------|----------------|------|---------|
| `{out}/bl.mha` | `uint8`/`int16` MetaImage (sitk zyx) | baseline CT | instance mask. Voxel = tracking id. Background `0`. |
| `{out}/fu.mha` | `uint8`/`int16` MetaImage (sitk zyx) | follow-up CT | instance mask. Voxel = tracking id. Background `0`. |
| `{out}/matches.csv` | UTF-8 CSV | — | pair table (see §1.4) |

Geometry = `props["sitk_stuff"]` (spacing/origin/direction) from that scan’s preprocess. Do not resample. Do not change BL voxel values.

Empty graph (no lesions on a side): that mask is all zeros; CSV still gets a header.

### 1.2 Who owns which id

- **BL mask is canonical.** Copy the BL instance mask as-is. Click `name` on BL **is** the tracking id.
- **FU mask is remapped.** Click names on FU are only used to build the matcher graph. After decode, every FU blob is painted with a tracking id from §1.3.
- Same tracking id on both masks = linked. Id only on BL = gone. Id only on FU = new.

This matches GT in `/nnunet_data/Longitudinal-CT/targetsTrBL` + `targetsTrFU`.

### 1.3 Paint rule (one function, all decode modes)

Decode only changes **which pairs exist**. Painting never branches on `dense` / `hungarian` / `sinkhorn`.

Pairs are `(bl_click_id, fu_click_id)` after decode. `bl_ids` / `fu_ids` = unique nonzero labels in the two instance masks.

```python
def fu_track_map(
    bl_ids: list[int],
    fu_ids: list[int],
    pairs: list[tuple[int, int]],
) -> dict[int, int]:
    """fu_click_id → tracking id painted on the FU mask."""
    fu_bls: dict[int, list[int]] = {}
    for b, f in pairs:
        fu_bls.setdefault(int(f), []).append(int(b))
    used = {int(x) for x in bl_ids}
    out: dict[int, int] = {}
    for f, bls in fu_bls.items():
        tid = min(bls)          # merge: lowest BL id; 1-to-1 / split: that BL id
        out[f] = tid
        used.add(tid)
    nxt = (max(used) + 1) if used else 1
    for f in fu_ids:
        f = int(f)
        if f in out:
            continue
        if f not in used:
            out[f] = f          # new: keep FU click id when free
            used.add(f)
            nxt = max(nxt, f + 1)
        else:
            out[f] = nxt        # collision with a BL id (incl. gone)
            used.add(nxt)
            nxt += 1
    return out
```

Apply:

```python
def paint_fu(fu_mask: np.ndarray, m: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(fu_mask, dtype=np.int32)
    for src, tid in m.items():
        out[fu_mask == src] = tid
    return out
```

`bl.mha` = BL instance mask copied (sitk zyx, uint8/int16). Never relabel BL.

#### What each decode does on the masks (consequence, not extra code)

| Decode | Pairs | Masks |
|--------|-------|-------|
| **`hungarian` (default)** | strict 1-to-1 | each FU blob gets at most one BL id. One id = one blob per mask. |
| **`sinkhorn`** | merges stay, splits drop | merged FU blob gets `min(BL ids)`. No two FU blobs share an id from a split. |
| **`dense`** | merges and splits stay | **merge:** FU blob ← `min(BL ids)`. Other BL ids stay on BL only (gone-into-merge). **split (option A):** every FU blob matched to that BL id is painted with that BL id → **one id, several disconnected blobs on FU**. |

Dense split is allowed to break “one id = one connected component” **on FU only**. That is the point of `--decode dense`. Document it. Do not try to keep CCs unique under dense.

Weird dense graph (one FU matched to several BL **and** one of those BL also matched to another FU): still `min` per FU blob. Do not add a second policy.

#### Worked numbers

**Hungarian.** BL labels `{1,2,5}`. FU click labels `{3,8,9}`. Pairs `1→3`, `2→8`. `5` unmatched (gone). `9` unmatched (new).

- map: `3→1`, `8→2`, `9→9` (9 not in used `{1,2,5}`)
- BL mask `{1,2,5}`, FU mask `{1,2,9}`

**Dense merge.** Pairs `2→8`, `5→8`.

- map: `8→min(2,5)=2`
- BL still `{2,5}`, FU blob is `2`

**Dense split (option A).** Pairs `1→3`, `1→9`.

- map: `3→1`, `9→1`
- BL `{1}`, FU two blobs both labeled `1`

**New-id collision.** BL `{1,3}` (3 gone). FU new click id `3`. `3` is in `used` → new tracking id `4`.

Ids are unique for **this** BL/FU volume pair. `_00` / `_01` body regions are separate CLI calls. Do not invent a patient-global namespace.

### 1.4 CSV

Keep the current four columns. Append `track_id` (the integer painted on the masks for that pair). Do not rename the old columns — `e2e_eval` and existing dumps stay valid.

```
bl_lesion_id,fu_lesion_id,pair_prob,decode,track_id
```

| Column | Value |
|--------|--------|
| `bl_lesion_id` | BL click / BL mask label (matcher node) |
| `fu_lesion_id` | FU **click** label **before** paint (matcher node) |
| `pair_prob` | `sigmoid` of that pair, unchanged |
| `decode` | `hungarian` / `dense` / `sinkhorn` |
| `track_id` | `min(BL ids matched to this fu_lesion_id)` = voxel value on `fu.mha` for that blob; also the voxel value on `bl.mha` for `bl_lesion_id` when it is the min |

One row per decoded pair. Header only if `pairs` is empty. Unmatched lesions are **not** extra CSV rows (same as today). They only appear as ids present on one mask.

`--pairs-out` (full N×M dump on `lesion_track`) stays as it is: `bl_lesion_id,fu_lesion_id,prob` with **click** ids, no `track_id`. Do not change it in Part 1.

### 1.5 Docs (write in the same change as the code, nanoUNet D1–D5)

Docs are part of the output contract. Stale docs = bug.

Touch these files when Part 2 lands. Do not write them before Part 2 is locked.

| File | What to change |
|------|----------------|
| `/nanoUNet/docs/steps/track.md` | 3-line summary, copy-paste command, **argument table (D3)**, inputs/outputs, errors. Must describe mask files + CSV + the table in §1.3. `<200` lines. |
| `/nanoUNet/docs/index.md` | Keep `predict → track` in the mermaid. Quickstart: one extra line that `nanounet_segtrack` writes `{bl,fu}.mha` + `matches.csv`. |
| `/nanoUNet/docs/reference/track_ids.md` | **New.** One concept: tracking ids on masks. The hungarian vs dense table from §1.3, the three worked numbers, “dense split = one id, several FU blobs”. No CLI flag dump (that lives in `steps/track.md`). `<200` lines. |
| `/nanoUNet/README.md` | If it lists CLIs, add `nanounet_segtrack` (today it omits it). |

`steps/track.md` outputs section must say, in this order:

1. `{out}/bl.mha` — BL instance mask, ids unchanged.
2. `{out}/fu.mha` — FU instance mask, ids remapped.
3. `{out}/matches.csv` — columns listed exactly as §1.4.

Errors table (E1) must include: missing `lesion-tracking`, missing ckpt / model-dir, BL/FU stem mismatch, missing sibling JSON, BL instance id with no FU-JSON point, empty instance mask (point at `docs/reference/track_ids.md` for “why FU ids ≠ click names”). This CLI defaults `--decode hungarian` — no TTY prompt.

Do not put this design essay in `technical.md`. That file is the matcher. Mask paint is a serving concern.

### 1.6 Non-goals for this output

- Do not encode merge/split in a second NIfTI or a color lookup.
- Do not write topology strings (`UNCHANGED` / …) onto voxels.
- Do not change how `instances_from_nifti` assigns click ids (still one CC ↔ one click id on the **input** to the matcher).
- Do not persist the pre-remap FU instance mask unless Part 2 adds an explicit debug flag. Default out dir has only the three files in §1.1.

---

## Part 2 — CLI + code

One command in nanoUNet. It imports `tracking` (lesion-tracking). No subprocess. No second repo CLI for this path. `lesion_track` stays CSV-only for people who already have instance masks.

Command name: **`nanounet_segtrack`** (already in `pyproject.toml`). Do not add `nanounet_seg_track`.

Flow per case, always:

```
BL CT+clicks → binary pred → instance
FU CT+clicks → binary pred → instance
track() → paint §1.3 → {bl,fu}.mha + matches.csv
```

Seg model: Dataset999 **single-stream**, two predict passes. Do **not** add `--longi`. Longi ckpts are FU-only and are a different product.

`--decode` default **`hungarian`**. No interactive decode prompt on this command.

---

### 2.1 Two input modes (mutually exclusive)

**Folder** (nnU-Net style, like `inputsTrFU`): sibling `{stem}.nii.gz` + `{stem}.json`. Pair by **exact stem** (`01161aaa0b_00` ↔ `01161aaa0b_00`), not by patient id alone (dual-region `_00`/`_01`).

```bash
nanounet_segtrack \
  --bl-dir /nnunet_data/Longitudinal-CT/inputsTrBL \
  --fu-dir /nnunet_data/Longitudinal-CT/inputsTrFU \
  --patients-csv /nnunet_data/Longitudinal-CT/test_patients.csv
```

**Single case:**

```bash
nanounet_segtrack \
  --bl-img /nnunet_data/Longitudinal-CT/inputsTrBL/01161aaa0b_00.nii.gz \
  --bl-clicks /nnunet_data/Longitudinal-CT/inputsTrBL/01161aaa0b_00.json \
  --fu-img /nnunet_data/Longitudinal-CT/inputsTrFU/01161aaa0b_00.nii.gz \
  --fu-clicks /nnunet_data/Longitudinal-CT/inputsTrFU/01161aaa0b_00.json
```

Argparse: one group, required. Passing `--bl-dir` without `--fu-dir` (or any single-mode flag without the other three) → `SystemExit` E1, not a traceback.

`--root` is **not** added. Two folders or four files. Nothing else.

Startup (R15), before any GPU work:

1. `import tracking` or E1 `pip install -e /lesion-tracking`
2. Folder: list `.nii.gz`, require sibling `.json`, intersect stems. BL-only or FU-only stems → E1 with up to 12 names + counts. Empty intersection → E1.
3. `--patients-csv` filters on `stem.split("_", 1)[0]` (same as `nanounet_predict`). Zero left → E1.
4. If `--meta` / `--meta-dir` set: those files must exist (types only, §2.3). If unset, do not look for `meta/`.
5. Model dir has `plans.json`, `dataset.json`, `nano_config.json`, ckpt. Track ckpt is a file.

---

### 2.2 Output paths

`NANOUNET_RESULTS` via `results_dir()`. Do not hardcode `/nnunet_data/NanoUNet_results` in the path join — the env already points there.

| Mode | `--out` omitted | `--out P` |
|------|-----------------|-----------|
| Folder | `{results}/segtrack/{fu_dir.name}/{stem}/` | `{P}/{stem}/` |
| Single | `{results}/segtrack/single/{stem}/` | `{P}/` (this **is** the case dir) |

`stem` = `01161aaa0b_00` (filename without `.nii.gz`). Each case dir is exactly §1.1 (`bl.mha`, `fu.mha`, `matches.csv`).

Example: `--fu-dir .../inputsTrFU` → `$NANOUNET_RESULTS/segtrack/inputsTrFU/01161aaa0b_00/fu.mha`. The extra `segtrack/` folder keeps these files out of `$NANOUNET_RESULTS/nanounet/` (training). The next folder is named after the input.

Resume: if `{case}/matches.csv` exists and `--overwrite` is off, skip the whole case (no re-predict). `cprint` dim skip line with stem.

`--keep-pred`: also write `{case}/pred_bl.mha` and `{case}/pred_fu.mha` (binary FG). Default **off**.

---

### 2.3 Points and optional type

**No meta CSV in the default path.** The FU click JSON is the propagated file.

On this dataset, `inputsTrFU/{stem}.json` points are already in follow-up space (`cog_propagated` for lesions that exist at BL; native FU clicks for new ones). `track()` already accepts that JSON (`load_propagated` → `_from_json`).

| Role | Source |
|------|--------|
| Seg prompts + instance ids | `--bl-clicks` / `--fu-clicks` (folder: sibling `.json`) |
| Matcher BL positions | **the FU JSON** (same file as `--fu-clicks`) |
| Matcher FU positions | mask centroids (unchanged) |
| Lesion type | optional meta; else `"unclear"` |

CLI always calls `track(..., propagated=fu_clicks_json)`. Do not add `--propagated` or `--prop-dir`.

**`load_propagated` must change.** Today it crashes unless the JSON names and the BL mask ids are the **same set**. Worked example:

- FU JSON names: `{1, 2, 3, 9}` (3 is gone on FU but still listed; 9 is new)
- BL pred mask after click-on-FG: `{1, 2}` (click 3 missed the predicted blob)

Today: error (`missing=[] extra=[3, 9]`). We cannot use the FU JSON as-is.

New rule:

- Every **BL mask** id must have a point in the JSON. `{1, 2}` ⊆ `{1, 2, 3, 9}` → ok. Look up those points, ignore 3 and 9 for BL positions.
- If the mask has id `5` and the JSON does not: still error (no coordinate for that BL lesion).

Replace `if want != got` with `missing = sorted(want - got); if missing: raise ...`. Extra names: do not error. Keep the rest of `propagate.py`.

**Lesion type is optional.** `track()` already defaults `default_lesion_type="unclear"`. JSON returns `typ={}`. That is enough.

Optional overlay, types **only** — do not read `cog_propagated` from meta:

| Flag | Mode | File |
|------|------|------|
| `--meta` | single | one CSV |
| `--meta-dir` | folder | `{meta_dir}/{pid}.csv`, `pid = stem.split("_", 1)[0]` |

New `load_types(path) -> dict[int, str]` in `tracking/data/propagate.py`: read `lesion_id` + `lesion_type`. Unknown type → E1 with `LESION_TYPES`. Add optional `types_csv: Path | None = None` to `track()` / `build_mask_graph`. After `load_propagated`, `typ.update(load_types(types_csv))` if set. If `--meta*` omitted, never open `meta/`.

---

### 2.4 Flags (locked)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--bl-dir` `--fu-dir` | path | — | Folder mode. Sibling `.nii.gz` + `.json` |
| `--bl-img` `--bl-clicks` `--fu-img` `--fu-clicks` | path | — | Single mode |
| `--meta` | path | unset | Single: optional types CSV (`lesion_id,lesion_type`). Not used for coordinates. |
| `--meta-dir` | path | unset | Folder: optional types CSVs `{pid}.csv`. Not used for coordinates. |
| `-o, --out` | path | §2.2 | Parent (folder) or case dir (single) |
| `-m, --model-dir` | path | §2.5 | Seg run dir (`plans.json` + ckpt) |
| `--ckpt` | str | `last.ckpt` | Seg checkpoint name |
| `--track-ckpt` | path | §2.5 | Matcher Lightning ckpt |
| `--decode` | choice | `hungarian` | `hungarian` / `dense` / `sinkhorn` |
| `--thresh` | float | `0.5` | Dense cutoff |
| `--device` | choice | `cuda` | `cuda` \| `cpu` \| `mps` |
| `--patients-csv` | path | unset | Folder filter |
| `--overwrite` | flag | off | Redo cases that already have `matches.csv` |
| `--keep-pred` | flag | off | Keep binary FG next to masks |
| `--ema` | flag | off | Seg EMA weights |
| `--batch-size` | int | `8` | Passed to `predict_case_logits` |
| `--inference-mode` | choice | `clustered` | `clustered` \| `centered` |
| `--disable-tta` | flag | config default | Same as predict |
| `--no-amp` | flag | off | |

Do not re-expose the rest of `nanounet_predict` (`--gt-dir`, `--longi`, `--baseline-*`, `--num-workers` as a user-facing predict-folder flag). Internally use **one** preprocess worker so FU CPU work overlaps BL GPU (G1). No nested progress bars.

---

### 2.5 Default models

Module constants (top of `nanounet/infer/segtrack.py`):

```python
DEFAULT_MODEL = Path(
    "/nnunet_data/NanoUNet_results/nanounet/"
    "Dataset999_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0_h200_instance_1200ep"
)
DEFAULT_TRACK = Path("/nnunet_data/lesion_tracking/runs/h60_r9/best.ckpt")
```

Resolve `-m`: cli → env `NANOUNET_SEGTRACK_MODEL` → `DEFAULT_MODEL`.  
Resolve `--track-ckpt`: cli → env `NANOUNET_SEGTRACK_TRACK` → `DEFAULT_TRACK`.

If the chosen path does not exist → E1 with the three ways to set it. No search, no second guess (R12).

`config_table` source column: `cli` / `env` / `default`.

---

### 2.6 Files (who writes what)

R1: every file `<200` LOC. R13: `cli/segtrack.py` is argparse + UI loop only.

| File | Role |
|------|------|
| `tracking/data/propagate.py` | Exact-match → subset (BL mask ids ⊆ JSON names). Add `load_types`. |
| `tracking/data/masks.py` | `build_mask_graph(..., types_csv=None)`: overlay `load_types` onto `typ`. |
| `tracking/data/paint.py` | **New.** `fu_track_map`, `paint_fu`, `write_case_masks`. Copy §1.3 code. Load BL/FU instance NIfTIs, write `bl.nii.gz` (copy BL, `int32`) and painted `fu.nii.gz`. Affine from those instance NIfTIs. ~90 LOC. |
| `tracking/infer.py` | `write_match_csv`: add column `track_id`. Compute map with `fu_track_map(...)`. Pass `types_csv` through `track()`. `propagated=` is the FU JSON when called from nanoUNet. Do not grow past 200; if tight, move CSV write into `paint.py`. |
| `nanounet/infer/segtrack.py` | **New.** `SegTrackCase` dataclass; `pair_folder`; `resolve_out`; `segment_native`; `run_case`. Imports existing `preprocess_case`, `predict_case_logits`, `export_prediction_from_logits`, `load_net_from_ckpt`, `pick_checkpoint`. Imports `instances_from_nifti`, `track`, `load_matcher`, `mask_has_lesions`, `write_match_csv`, paint helpers. |
| `nanounet/cli/segtrack.py` | **Replace** current pred-in CLI. `main()`: `quiet_lightning_runtime()`, parse, validate, banner, load nets **once**, progress loop, summary panel. |
| `nanounet/common.py` | Add `nano_banner(title, subtitle, color="cyan")` and `console() -> Console`. Do **not** change `nano_progress` yield type (other CLIs depend on it). ~15 lines. |
| `nanounet/cli/__init__.py` | Docstring already lists segtrack. No extra text. |
| Docs | §1.5, same change. |

Do not add `nanounet/cli/segtrack_ui.py`. Do not add a `utils/` folder. Do not wrap Lightning.

`segment_native` is the one-scan predict, copy the body of predict.py `gpu()` without scoring:

```python
def segment_native(net, lm, cfg, pl, cm, dj, dev, scan: Path, clicks: Path, out_nii: Path, *,
                   use_tta, border_expand, max_border_extra, batch_size, use_amp,
                   cluster_margin_frac, inference_mode, no_prompt_encode=False) -> None:
    pack = preprocess_case(str(scan), str(clicks), pl, cm, dj, None, None)
    pad_cpu, slicer_revert, props, points_xyz, bl_points = pack
    pad = pad_cpu.pin_memory().to(dev, non_blocking=True) if dev.type == "cuda" else pad_cpu.to(dev)
    logits, tiles = predict_case_logits(
        net=net, lm=lm, cfg=cfg, pl=pl, cm=cm, dev=dev,
        pad=pad, slicer_revert=slicer_revert, props=props, points_xyz=points_xyz,
        encode_prompt=not no_prompt_encode, use_tta=use_tta,
        border_expand=border_expand, max_border_expand_extra=max_border_extra,
        batch_size=batch_size, use_amp=use_amp,
        cluster_margin_frac=cluster_margin_frac, mode=inference_mode,
        is_longi=False, bl_present=False, bl_points_xyz=bl_points,
    )
    out_trunc = str(out_nii)
    end = dj["file_ending"]
    if out_trunc.endswith(end):
        out_trunc = out_trunc[: -len(end)]
    export_prediction_from_logits(logits, props, cm, pl, dj, out_trunc, tiles)
```

`run_case` (order is fixed):

1. `case_dir.mkdir`
2. If not overwrite and `matches.csv` exists: return `"skip"`
3. Temp dir: `segment_native` → `pred_bl`, `pred_fu` (overlap: start FU `preprocess_case` in a `ThreadPoolExecutor(max_workers=1)` **while** BL is on GPU; then `gpu` FU)
4. `instances_from_nifti` both
5. If `not mask_has_lesions(bl_inst) and not mask_has_lesions(fu_inst)`: write all-zero masks (copy geometry from instance niftis) + CSV header only; return `"empty"`. Do **not** call `track()` (`build_mask_graph` asserts nonempty labels).
6. If only one side empty: same — CSV header, copy the nonempty instance as BL or paint-empty FU; no `track()`.
7. Else `track(..., propagated=fu_clicks, types_csv=meta_or_none, matcher=matcher, decode=decode, ...)` then `write_case_masks` + `write_match_csv`
8. Delete temp unless `--keep-pred` (then copy binaries into `case_dir`)
9. Return `"ok"` plus `n_pairs`, seconds

Load **once** in `main`, pass in:

```python
net, lm = load_net_from_ckpt(pick_checkpoint(model_dir, ckpt), cm, dj, dev, longi=False, ema=ema)
matcher = load_matcher(track_ckpt, device)
```

Do not reload per case. Do not `subprocess` `nanounet_predict`.

---

### 2.7 UX (Vite / Claude-Code calm, not a screensaver)

Rich only. One `Console` (`common.console()`), stderr. No extra `print`, no `tqdm`, no emoji, no ASCII animation, no color cycle.

**Open** (before config table):

```python
nano_banner("nanoUNet  seg × track", "scans + clicks → linked instance masks")
```

Implement `nano_banner` as a `Panel` with `Align.center`, `padding=(1, 4)`, `border_style=color`. Title bold cyan, subtitle dim. That is the whole wow. Stop there.

Then `config_table` with rows: model-dir, ckpt, track-ckpt, decode, device, n_cases, out. Sources `cli`/`env`/`default`.

Then **one** `Progress` (U4): `SpinnerColumn`, description, `BarColumn`, `completed/total`, `TimeElapsedColumn`, `console=console()`, `transient=False` (bar stays; this is the live view). Description format, update in place, no extra per-case green lines (U7):

```
12/60  03b90eb112_00  ·  segment BL
12/60  03b90eb112_00  ·  segment FU
12/60  03b90eb112_00  ·  track
```

On skip/empty, set description to `skip` / `empty` and advance.

**Close** with a `Panel` (green border):

```
60 cases  ·  53 linked  ·  4 empty  ·  3 skip
412 pairs  ·  14m 02s
wrote  /nnunet_data/NanoUNet_results/segtrack/inputsTrFU
next   open fu.mha — same integer = same lesion
       docs/reference/track_ids.md
```

`cprint` the suggested next command only in that panel, not again below.

Lightning/wandb: `quiet_lightning_runtime()` first line of `main`, before importing PL via tracking.

---

### 2.8 Errors (copy these)

Missing tracking:

```
tracking is not installed.
Expected the lesion-tracking package on PYTHONPATH.
Fix: pip install -e /lesion-tracking
```

Stem mismatch:

```
BL/FU folders do not share the same case names.
--bl-dir has 12 stems not in --fu-dir (e.g. aaa_00, bbb_01, ...).
--fu-dir has 3 stems not in --bl-dir (e.g. ccc_00).
Fix: pass matching inputsTrBL and inputsTrFU, or --patients-csv to select a subset
(see docs/steps/track.md)
```

Missing default model:

```
No seg model at {path}.
Expected a nanoUNet run dir with plans.json and checkpoints/last.ckpt.
Fix: nanounet_segtrack -m $NANOUNET_RESULTS/nanounet/<run>   or export NANOUNET_SEGTRACK_MODEL=...
(see docs/steps/track.md)
```

User mistakes: `SystemExit` / `FileNotFoundError` with those three lines. No 40-frame stack (E5).

---

### 2.9 Docs (same change as code)

Rewrite `/nanoUNet/docs/steps/track.md` to this CLI (folder example first, then single). Argument table = §2.4. Outputs = §1.1 + §2.2. Link `docs/reference/track_ids.md`. Commands must be the literals in §2.1 (D5). File `<200` lines.

`docs/index.md`: mermaid stays `predict → track`. Quickstart add the folder `nanounet_segtrack` command.

`README.md`: list `nanounet_segtrack` next to predict.

New `docs/reference/track_ids.md` = §1.5.

---

### 2.10 Build order (after sign-off)

1. `load_propagated` subset + `load_types`. Tiny array smoke for `fu_track_map` (the three worked numbers in §1.3). Delete after pass (R16).
2. `tracking/data/paint.py` + `write_match_csv` column.
3. `nanounet/infer/segtrack.py` + `common.py` banner/console.
4. Replace `nanounet/cli/segtrack.py`.
5. Docs.
6. One real case: `01161aaa0b_00` from Longitudinal-CT, `--decode hungarian`, **no meta**, confirm `bl.mha`/`fu.mha`/`matches.csv` exist and shared ids ⊂ BL labels.

Do not run the 59-patient holdout in this change.

---

### 2.11 Non-goals

- Dataset114 longi two-stream predict
- Requiring or auto-loading `meta/*.csv` for coordinates
- Changing `lesion_track` into a folder predictor
- Writing pre-remap instance masks
- Interactive decode menu
- New dependencies

---

## Part 3 — BL instance mask input

Optional. Omit both flags → Part 2 two-UNet path.

| Flag | Mode | Meaning |
|------|------|---------|
| `--bl-mask` | single | Native BL instance NIfTI/`.mha`. Voxel = tracking id. |
| `--bl-mask-dir` | folder | `{stem}.nii.gz` (fallback `.mha`) for every paired stem. |

XOR. `--bl-clicks` forbidden with `--bl-mask`. Folder mask mode: BL dir is CT-only (no sibling JSON required). FU JSON still required (matcher BL coordinates). BL CT still required (L0).

Per case: load BL mask sitk zyx → skip BL UNet → predict FU → click-CC FU → `track()` → `paint_fu`. `bl.mha` copies given labels (no relabel). `--keep-pred` writes `pred_fu.mha` only.

No `lesion-tracking` changes. No `--gt-dir` scoring.

