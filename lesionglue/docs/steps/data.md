# Data

Turn the raw Longitudinal-CT dataset into the cached graphs the matcher trains on, and audit the raw labels.
`lesionglue_split` writes the tracking split, `lesionglue_preprocess` builds cached L0 graphs for it, `lesionglue_audit` reports label statistics.
Needs `data_split.json`, `test_patients.csv`, `meta/{patient}.csv` and the NIfTIs under the dataset root.

## lesionglue_split

Carve train/val from the official train patients (patient-level folds) and take the 60 patients in `test_patients.csv` as test.

### Command

```bash
lesionglue_split --root /nnunet_data/Longitudinal-CT --holdout /nnunet_data/Longitudinal-CT/test_patients.csv --out lesionglue/configs/split.json
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--root` | path | `/nnunet_data/Longitudinal-CT` | Longitudinal-CT dataset root holding `data_split.json` (official train/val/test) |
| `--holdout` | path | `/nnunet_data/Longitudinal-CT/test_patients.csv` | CSV of held-out test patient ids (column `patient`); these become the test split |
| `--out` | path | `lesionglue/configs/split.json` | Output path for the tracking split JSON |
| `--n-folds` | int | 5 | Number of patient-level folds carved from the official train patients |
| `--seed` | int | 0 | RNG seed for the patient-to-fold assignment |
| `--val-fold` | int | 0 | Fold index (0-based) used as val; the other folds form train |

### Inputs / outputs

| Path | Format | Written by |
|---|---|---|
| `/nnunet_data/Longitudinal-CT/data_split.json` | JSON (`train`, `val`, `test`) | you |
| `/nnunet_data/Longitudinal-CT/test_patients.csv` | CSV (`patient` column) | you |
| `lesionglue/configs/split.json` | JSON (`train`, `val`, `test`) | this step |

### Common errors

| Message starts with | Fix |
|---|---|
| `No official split at` | `lesionglue_split --root /nnunet_data/Longitudinal-CT` (the dir must hold `data_split.json`) |
| `No holdout CSV at` | `--holdout /nnunet_data/Longitudinal-CT/test_patients.csv` |
| `Empty holdout CSV` | Add a header plus one patient id per row to the `--holdout` file |
| `No patient ids in` | Fill the `patient` column of the `--holdout` file |
| `N holdout ids in official train` | `--holdout` lists patients that are in the official train set; use the matching `test_patients.csv` |
| `official val/test id not in holdout csv` | The official val/test patients must all be in `--holdout` |
| `holdout id not in official val∪test` | Every `--holdout` id must be in the official val or test split |

## lesionglue_preprocess

Build cached dense `v8_native` PyG graphs (L0 descriptors) for one tracking split, or all three. Reads the split JSON written by `lesionglue_split`.

### Command

```bash
lesionglue_preprocess --split all --root /nnunet_data/Longitudinal-CT --cache /nnunet_data/lesion_tracking/cache --jobs 16
```

Fill the 113 BL lesions that lack `cog_propagated` from the uniGradICON registration (only where its `sanity_ok` is true; the other 16 stay dropped), in a cache of its own. `--keep-unclear` keeps the 135 `linking_unclear` lesions (default drops them) for the unclear-link sensitivity row:

```bash
lesionglue_preprocess --split all --prop-fill unigradicon --cache /nnunet_data/lesion_tracking/cache_v9 --jobs 4
lesionglue_preprocess --split all --prop-fill unigradicon --keep-unclear --cache /nnunet_data/lesion_tracking/cache_v9_unclear --jobs 4
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--root` | path | `/nnunet_data/Longitudinal-CT` | Longitudinal-CT dataset root holding the raw cases |
| `--cache` | path | `/nnunet_data/lesion_tracking/cache` | Output root for the cached graphs (written under `processed/`) |
| `--split` | choice | required | `train`, `val`, `test`, or `all` (builds train, val and test) |
| `--k-intra` | int | 8 | Neighbors per node in the intra-timepoint kNN graph |
| `--jobs` | int | 1 | Parallel patients (ProcessPool); each worker pins BLAS/OpenMP to 1 thread |
| `--keep-unclear` | flag | off | Keep lesions whose `linking_unclear` flag is set (the default drops those rows). Needs its own `--cache` dir, and every split of one cache dir must use the same setting |
| `--prop-fill` | choice | `none` | `none`: a BL lesion without `cog_propagated` (129 in 25 patients) gets no node. `unigradicon`: it gets the registration's `bl_click` from `derivatives/unigrad-icon-registration/` where `sanity_ok` is true (`prop_source` 1 on that node); the rest stay dropped. Same cache-dir rule as `--keep-unclear` |
| `--resume` | flag | off | Keep already-built patients in the staging dir and build only the missing ones, then merge all patients at the end; default rebuilds the split |

### Inputs / outputs

| Path | Format | Written by |
|---|---|---|
| `lesionglue/configs/split.json` | JSON | `lesionglue_split` |
| `/nnunet_data/Longitudinal-CT/{meta,inputsTrBL,inputsTrFU,targetsTrBL,targetsTrFU}/` | CSV, NIfTI | you |
| `/nnunet_data/lesion_tracking/cache/processed/staging/{split}_v8_native/{patient}.pt` | PyG graphs, one file per patient | this step |
| `/nnunet_data/lesion_tracking/cache/processed/{split}_v8_native.pt` | collated PyG dataset | this step |
| `/nnunet_data/lesion_tracking/cache/processed/{split}_v8_native_meta.pt` | edge/positive counts, descriptor dims (`keep_unclear`, `prop_fill` when not default) | this step |

Old `*_v5_l0.pt` files in `processed/` are ignored (a warning is printed); delete them if they confuse you.

### Common errors

| Message starts with | Fix |
|---|---|
| `No tracking split at` | `lesionglue_split --root /nnunet_data/Longitudinal-CT` first |
| `split '...' missing from the split file` | Regenerate `lesionglue/configs/split.json` with `lesionglue_split` (needs keys train/val/test) |
| `would write into the default cache` | `--cache /nnunet_data/lesion_tracking/cache_v9` (a dir of its own for `--keep-unclear` / `--prop-fill`) |
| `was built with keep_unclear=` or `was built with prop_fill=` | The cache dir already holds splits built with other settings; match the flag to it or use a new `--cache` dir |
| `merge target(s) [...] have no FU node` | A MERGING row needs a row with `lesion_id == merged_into` and a `cog_fu` in `meta/{patient}.csv` |
| `zero graphs for split=` | The split has no buildable patients; check `--root` and `lesionglue_split` output |
| `empty mask for lesion` | A lesion id in `meta/{patient}.csv` has no voxels in its mask; fix that patient's masks or CSV |
| `unknown topology_class` | `meta/{patient}.csv` has a topology label outside the known set; fix the CSV |
| `unknown lesion_type` | `meta/{patient}.csv` has a lesion type outside the known set; fix the CSV |

## lesionglue_audit

Read-only label audit for one split of the official `data_split.json`: class balance, imputation rates, lesion-size quartiles. With `--ckpt` it also stratifies matching errors by registration provenance (imputed vs observed). Trains nothing.

### Command

```bash
lesionglue_audit --root /nnunet_data/Longitudinal-CT --cache /nnunet_data/lesion_tracking/cache --split val --out runs/audit
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--root` | path | `/nnunet_data/Longitudinal-CT` | Longitudinal-CT dataset root (`data_split.json`, `meta/`, `inputsTrBL/`, ...) |
| `--cache` | path | `/nnunet_data/lesion_tracking/cache` | Cached lesion-graph dir; only read with `--ckpt` (error stratification) |
| `--split` | choice | `val` | Which `data_split.json` split to audit: `train`, `val`, or `test` |
| `--out` | path | `runs/audit` | Output dir for `audit_<split>.json` (created if missing) |
| `--ckpt` | path | none | If given, run section 4 (error stratification) |
| `--no-ema` | flag | off | With `--ckpt`: score the raw weights instead of the EMA weights |

### Inputs / outputs

| Path | Format | Written by |
|---|---|---|
| `/nnunet_data/Longitudinal-CT/data_split.json` | JSON | you |
| `/nnunet_data/Longitudinal-CT/meta/{patient}.csv` | CSV | you |
| `/nnunet_data/lesion_tracking/cache/processed/{split}_v8_native.pt` | PyG dataset, read only with `--ckpt` | `lesionglue_preprocess` |
| `runs/audit/audit_{split}.json` | JSON (class balance, imputation, quartiles, optional stratification) | this step |

### Common errors

| Message starts with | Fix |
|---|---|
| `[Errno 2] No such file or directory` | `--root` has no `data_split.json` (or a patient's `meta/{patient}.csv` is missing); point `--root` at `/nnunet_data/Longitudinal-CT` |
| `unknown topology_class` | `meta/{patient}.csv` has a topology label outside the known set; fix the CSV |
| `unknown lesion_type` | `meta/{patient}.csv` has a lesion type outside the known set; fix the CSV |
