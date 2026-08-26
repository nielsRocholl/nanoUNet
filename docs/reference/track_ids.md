# Tracking ids on masks

Same integer on both scans = same lesion. Background is `0`. BL click names are canonical; the FU mask is remapped after matching. Decode only changes which pairs exist; painting is one function.

| Decode | Pairs | Masks |
|--------|-------|-------|
| **hungarian** (default) | strict 1-to-1 | each FU blob gets at most one BL id. One id = one blob per mask. |
| **sinkhorn** | merges stay, splits drop | merged FU blob gets `min(BL ids)`. |
| **dense** | merges and splits stay | **merge:** FU blob ← `min(BL ids)`. Other BL ids stay on BL only. **split:** every FU blob matched to that BL id is painted with that BL id → **one id, several disconnected blobs on FU**. |

Dense split is allowed to break “one id = one connected component” **on FU only**. That is the point of `--decode dense`.

CLI: [steps/track.md](../steps/track.md). Files: `{out}/bl.nii.gz`, `{out}/fu.nii.gz`, `{out}/matches.csv`.

## Worked numbers

**Hungarian.** BL `{1,2,5}`. FU click `{3,8,9}`. Pairs `1→3`, `2→8`. `5` gone, `9` new.

- map: `3→1`, `8→2`, `9→9`
- BL mask `{1,2,5}`, FU mask `{1,2,9}`

**Dense merge.** Pairs `2→8`, `5→8`. FU blob is `2`. BL still `{2,5}`.

**Dense split.** Pairs `1→3`, `1→9`. Both FU blobs labeled `1`.

**New-id collision.** BL `{1,3}` (3 gone). FU new click id `3` → tracking id `4`.

FU click names are **not** the ids on `fu.nii.gz` after paint. Join CSV `fu_lesion_id` (pre-paint click) to `track_id` (voxel value).
