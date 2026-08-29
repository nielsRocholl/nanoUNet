# segtrack instance labels, tracking IDs, and EMA

Status: implementation-ready. Complete this document before coordinate or registration work.

## 1. Preserve every predicted component

Current `/lesion-tracking/tracking/data/instances.py::binary_to_instances` labels only
components hit by an exact click. Unclaimed components remain zero in the LUT and disappear.
Held-out EMA predictions contain 687 components; current labeling retains 311 and deletes
376. Of 549 GT lesions, 135 have a deleted best-overlap component with Dice >= 0.5.

Replace `binary_to_instances` with:

```python
def label_instances(pred: np.ndarray, clicks_zyx: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """Label every 18-connected component; exact click hits keep their lesion ID."""
    lab = cc3d.connected_components((pred > 0).astype(np.uint8), connectivity=18)
    n = int(lab.max())
    if n == 0:
        return lab.astype(np.int32)
    lut = np.zeros(n + 1, dtype=np.int32)
    claimed: dict[int, int] = {}
    conflicts: list[tuple[int, int, int]] = []
    for lid, (z, y, x) in clicks_zyx.items():
        z = min(max(int(z), 0), pred.shape[0] - 1)
        y = min(max(int(y), 0), pred.shape[1] - 1)
        x = min(max(int(x), 0), pred.shape[2] - 1)
        cc = int(lab[z, y, x])
        if cc == 0:
            continue
        if cc in claimed and claimed[cc] != lid:
            conflicts.append((lid, claimed[cc], cc))
            continue
        claimed[cc] = lid
        lut[cc] = lid
    if conflicts:
        cprint(f"[yellow]click CC conflicts (later click skipped): {conflicts}[/yellow]")
    next_id = max(clicks_zyx, default=0) + 1
    for cc in range(1, n + 1):
        if lut[cc] == 0:
            lut[cc] = next_id
            next_id += 1
    return lut[lab]
```

Fresh IDs start above every click ID, including a canonical ID lost in a click conflict.
Do not add nearest-voxel snapping. Do not claim this separates touching lesions: 38 held-out
clicks collide with a component already claimed by another click. Fixing that requires
instance-aware model output and is outside this change.

Update:

- `instances_from_nifti` to call `label_instances`.
- `/nanoUNet/nanounet/infer/segtrack.py` import.
- FU call in the BL-mask branch.
- BL and FU calls in the predicted-BL branch.

Do not alter `load_clicks` or the BL-mask load path.

## 2. Reserve every BL ID when painting FU

`track()` returns only BL IDs admitted to the graph. Passing `r.bl_ids` into `fu_track_map`
allows an unmatched FU ID to reuse a BL ID excluded for missing coordinates. Then equal mask
integers falsely imply a match.

In `run_case`, compute all positive BL mask labels once:

```python
mx = int(bl_zyx.max())
all_bl_ids = (
    np.flatnonzero(np.bincount(bl_zyx.ravel(), minlength=mx + 1))[1:].tolist()
    if mx > 0 else []
)
```

Reuse `all_bl_ids` for coordinate coverage checks. Change only the final paint-map call:

```python
pairs = [(int(r.bl_ids[i]), int(r.fu_ids[j])) for i, j in r.pairs]
m = fu_track_map(all_bl_ids, list(map(int, r.fu_ids)), pairs)
```

Keep `write_match_csv` unchanged: its rows contain matched graph nodes only.

Required invariant after writing masks and CSV:

```python
shared = set(np.unique(bl_zyx)) & set(np.unique(fu_out))
matched = {int(row["track_id"]) for row in csv.DictReader(csv_path.open())}
assert shared - {0} <= matched
```

Use this in a throwaway verification script, not production hot-path code.

## 3. Default segmentation to EMA without breaking `--ema`

The deployed segmentation checkpoint contains 956 raw `net.*` tensors and a matching
956-tensor `callbacks/EMACallback/shadow`. EMA is therefore available.

In `/nanoUNet/nanounet/cli/segtrack.py::_mode`, replace `store_true` with:

```python
ap.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
```

Python >=3.10 is required by `pyproject.toml`, so this provides both `--ema` and
`--no-ema`. Keeping `--ema` preserves existing commands.

Add the resolved row:

```python
("seg-ema", "on" if args.ema else "off", "cli" if not args.ema else "default"),
```

Leave matcher EMA and `load_net_from_ckpt(..., ema=args.ema)` unchanged. Do not change
`nanounet_predict`.

`/nanoUNet/nanounet/infer/predictor.py::_ema_shadow` is shared by both CLIs. Replace both
messages that say `drop --ema` with:

```text
Fix: pass --no-ema in nanounet_segtrack, or omit --ema in nanounet_predict
```

Update the `docs/steps/track.md` argument row to `--ema / --no-ema`, default `on`.

## 4. Verification

No UNet run:

1. Read the four EMA predictions named below.
2. Run `label_instances`.
3. Assert `count_nonzero(unique(inst)) == connected_components(...).max()`.

Expected component counts:

- `307fd7f231_00`: 2; exact click hits 0.
- `38b18881fc_00`: 13; exact click hits 10.
- `bf97f24695_00`: 25; unique claimed CCs 10; 9 extra clicks collide on an already-claimed CC.
- `0f49c89d1e_00`: 8; exact click hits 2.

Real matcher smoke test using EMA prediction + GT BL mask for `307fd7f231_00`:

- provisional FU IDs become 3 and 4;
- deployed matcher pairs BL 2 to FU 4;
- final lesion-2 Dice is approximately 0.893;
- lesion 1 remains unmatched, proving Phase 1 is not a complete metric fix.

CLI gate:

```bash
nanounet_segtrack \
  --bl-img /nnunet_data/Longitudinal-CT/inputsTrBL/307fd7f231_00.nii.gz \
  --bl-mask /nnunet_data/Longitudinal-CT/targetsTrBL/307fd7f231_00.nii.gz \
  --fu-img /nnunet_data/Longitudinal-CT/inputsTrFU/307fd7f231_00.nii.gz \
  --fu-clicks /nnunet_data/Longitudinal-CT/inputsTrFU/307fd7f231_00.json \
  -o "$NANOUNET_RESULTS/segtrack/phase1_check/307fd7f231_00" \
  --overwrite --keep-pred
```

Assert `fu.mha > 0` equals `pred_fu.mha > 0`, config shows `seg-ema on`, and
`--no-ema` shows `seg-ema off`. Delete throwaway scripts after validation.
