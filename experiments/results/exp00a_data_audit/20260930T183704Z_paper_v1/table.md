# exp00a data audit

## Claims (paper value vs measured)

| claim | paper_value | measured_value | match | source |
|---|---|---|---|---|
| Longitudinal-CT patients | 300 | 300 | OK | meta/*.csv |
| linked lesions, one scan pair per patient | 4530 | 4530 | OK | Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept |
| unchanged lesions | 2424 | 2424 | OK | Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept |
| disappearing lesions | 1382 | 1382 | OK | Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept |
| newlyappearing lesions | 558 | 558 | OK | Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept |
| merging lesions | 166 | 166 | OK | Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept |
| topology labels per lesion (distinct values) | 4 | 4 | OK | Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept |
| merge events | 38 | 38 | OK | Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept, groups of MERGING rows by merged_into |
| merge events of two lesions into one | 24 | 24 | OK | Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept |
| split labels (SPLIT rows) | 0 | 0 | OK | meta/*.csv topology_class (all rows) |
| held-out patients | 60 | 60 | OK | test_patients.csv |
| held-out = released val (30) + test (30) | 60 | 60 | OK | data_split.json vs test_patients.csv |
| released split: validation patients | 30 | 30 | OK | data_split.json |
| released split: test patients | 30 | 30 | OK | data_split.json |
| remaining training pool patients | 240 | 240 | OK | data_split.json |
| held-out lesions | 774 | 774 | OK | Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept, held-out 60 |
| held-out unchanged lesions | 475 | 475 | OK | Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept, held-out 60 |
| held-out disappearing lesions | 213 | 213 | OK | Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept, held-out 60 |
| held-out newlyappearing lesions | 67 | 67 | OK | Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept, held-out 60 |
| held-out merging lesions | 19 | 19 | OK | Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept, held-out 60 |
| held-out merge events | 5 | 5 | OK | Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept, held-out 60 |
| held-out merge events are in five patients | 5 | 5 | OK | Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept, held-out 60 |
| held-out largest merge: nine lesions into one | 9 | 9 | OK | Longitudinal-CT/meta/*.csv, dominant img_id_fu, linking_unclear kept, held-out 60 |
| tab:datasets volumes, Longitudinal-CT | 537 | 537 | OK | splits_final.json train+val, prefix d013 |
| tab:datasets volumes, LIDC-IDRI | 897 | 897 | OK | splits_final.json train+val, prefix d024 |
| tab:datasets volumes, LNDb | 216 | 216 | OK | splits_final.json train+val, prefix d012 |
| tab:datasets volumes, MSD Lung | 63 | 63 | OK | splits_final.json train+val, prefix d015 |
| tab:datasets volumes, RIDER Lung CT | 59 | 59 | OK | splits_final.json train+val, prefix d029 |
| tab:datasets volumes, Mediastinal LN | 120 | 120 | OK | splits_final.json train+val, prefix d030 |
| tab:datasets volumes, MCT-LTDiag | 516 | 516 | OK | splits_final.json train+val, prefix d027 |
| tab:datasets volumes, WAW-TACE | 232 | 232 | OK | splits_final.json train+val, prefix d019 |
| tab:datasets volumes, CRLM | 197 | 197 | OK | splits_final.json train+val, prefix d011 |
| tab:datasets volumes, MSD Liver | 130 | 130 | OK | splits_final.json train+val, prefix d017 |
| tab:datasets volumes, LiTS | 126 | 126 | OK | splits_final.json train+val, prefix d023 |
| tab:datasets volumes, WORC CRLM | 77 | 77 | OK | splits_final.json train+val, prefix d020 |
| tab:datasets volumes, PanTS | 880 | 880 | OK | splits_final.json train+val, prefix d028 |
| tab:datasets volumes, MSD Pancreas | 281 | 281 | OK | splits_final.json train+val, prefix d016 |
| tab:datasets volumes, RUMC Pancreas | 13 | 13 | OK | splits_final.json train+val, prefix d026 |
| tab:datasets volumes, KiTS23 | 489 | 489 | OK | splits_final.json train+val, prefix d022 |
| tab:datasets volumes, MSWAL | 284 | 284 | OK | splits_final.json train+val, prefix d018 |
| tab:datasets volumes, WORC GIST | 245 | 245 | OK | splits_final.json train+val, prefix d021 |
| tab:datasets volumes, MSD Colon | 126 | 126 | OK | splits_final.json train+val, prefix d014 |
| tab:datasets volumes, Adrenal-ACC-Ki67 | 51 | 51 | OK | splits_final.json train+val, prefix d031 |
| tab:datasets volumes, RUMC Bone | 151 | 151 | OK | splits_final.json train+val, prefix d025 |
| tab:datasets total volumes | 5690 | 5690 | OK | splits_final.json train+val |
| 5690 volumes: cohorts.json total | 5690 | 5690 | OK | cohorts.json counts |
| 5690 volumes: dataset.json entries | 5690 | 5796 | MISMATCH | dataset.json |
| 21 cohorts (corpus) | 21 | 21 | OK | splits_final.json prefixes |
| 21 cohorts (NanoUNet_raw dirs Dataset011..031) | 21 | 21 | OK | NanoUNet_raw listing |
| 15 % of each cohort held out for validation (pooled share, %) | 15 | 15.061511423550089 | OK | splits_final.json |
| 15 % of each cohort held out for validation (every cohort within one case of 15 %) | 0 | 1 | MISMATCH | splits_final.json |
| Longitudinal-CT enters the corpus as 537 separate scans | 537 | 537 | OK | splits_final.json d013 |
| ... from 240 patients | 240 | 240 | OK | splits_final.json d013 case names |
| ... the 240 are the released split's training patients | 240 | 240 | OK | data_split.json train |
| ... none of the held-out 60 is in the corpus | 0 | 0 | OK | splits_final.json d013 vs test_patients.csv |
| median lesion radius (equivalent sphere, vox) in the pooled corpus | 4.6 | 4.638339757884534 | OK | centroids.json volume_vox, all lesions of the 5690 volumes |
| neighbour within 30 vox for x % of lesions | 42 | 32.955487973458666 | MISMATCH | centroids.json centroids_zyx, nearest other lesion, Euclidean voxels, all lesions |
| PanTrack patients | 45 | 45 | OK | PanTrack patients/tracking/organ_annotations.json |
| PanTrack CT scans | 161 | 161 | OK | patients.json |
| PanTrack label files | 161 | 161 | OK | labels/ listing |
| PanTrack annotated lesion instances | 292 | 289 | MISMATCH | organ_annotations.json entries |
| PanTrack annotated lesion instances (label NIfTIs) | 292 | 289 | MISMATCH | labels/*.nii.gz distinct non-zero values per scan |
| PanTrack pancreas instances | 165 | 164 | MISMATCH | organ_annotations.json |
| PanTrack liver instances | 124 | 122 | MISMATCH | organ_annotations.json |
| PanTrack lymph-node instances | 3 | 3 | OK | organ_annotations.json |
| PanTrack consecutive scan pairs | 116 | 116 | OK | tracking.json |
| PanTrack vanishing lesion pairs | 36 | 36 | OK | tracking.json fu_point null |
| PanTrack propagated point for every lesion (count = tracked lesion entries) | 217 | 217 | OK | tracking.json fu_point_prop |

## linking_unclear reconciliation (paper rule = unclear kept)

| scope | view | n_patients | n_lesions | UNCHANGED | DISAPPEARING | NEWLYAPPEARING | MERGING |
|---|---|---|---|---|---|---|---|
| all300 | paper_rule_unclear_kept | 300 | 4530 | 2424 | 1382 | 558 | 166 |
| all300 | unclear_excluded | 300 | 4396 | 2421 | 1251 | 558 | 166 |
| all300 | unclear_only | 300 | 134 | 3 | 131 | 0 | 0 |
| official_train240 | paper_rule_unclear_kept | 240 | 3756 | 1949 | 1169 | 491 | 147 |
| official_train240 | unclear_excluded | 240 | 3637 | 1946 | 1053 | 491 | 147 |
| official_train240 | unclear_only | 240 | 119 | 3 | 116 | 0 | 0 |
| official_val30 | paper_rule_unclear_kept | 30 | 335 | 199 | 109 | 19 | 8 |
| official_val30 | unclear_excluded | 30 | 321 | 199 | 95 | 19 | 8 |
| official_val30 | unclear_only | 30 | 14 | 0 | 14 | 0 | 0 |
| official_test30 | paper_rule_unclear_kept | 30 | 439 | 276 | 104 | 48 | 11 |
| official_test30 | unclear_excluded | 30 | 438 | 276 | 103 | 48 | 11 |
| official_test30 | unclear_only | 30 | 1 | 0 | 1 | 0 | 0 |
| holdout60 | paper_rule_unclear_kept | 60 | 774 | 475 | 213 | 67 | 19 |
| holdout60 | unclear_excluded | 60 | 759 | 475 | 198 | 67 | 19 |
| holdout60 | unclear_only | 60 | 15 | 0 | 15 | 0 | 0 |

## Graph cache replay vs cache metadata

| split | n_patients_in_split | n_patients_with_graph | n_graphs | n_patients_multi_graph | edges_replayed | positives_replayed | edges_cache_meta | positives_cache_meta | matches_cache_meta |
|---|---|---|---|---|---|---|---|---|---|
| train | 192 | 180 | 199 | 17 | 55410 | 1659 | 55410 | 1659 | True |
| val | 48 | 46 | 48 | 2 | 7806 | 340 | 7806 | 340 | True |
| test | 60 | 57 | 59 | 2 | 11366 | 462 | 11366 | 462 | True |

## Patients with no graph in the cache

| patient | graph_split | why | n_bl_pair | n_fu_pair | n_unclear_pair | in_holdout60 |
|---|---|---|---|---|---|---|
| 06eb133bbf | train | FU region 0: 2 BL nodes, 0 FU nodes | 2 | 0 | 0 | False |
| 06eb61b839 | train | FU region 0: 0 BL nodes, 9 FU nodes | 8 | 9 | 0 | False |
| 0777d5c17d | train | FU region 0: 1 BL nodes, 0 FU nodes | 1 | 0 | 0 | False |
| 0aa1883c64 | train | FU region 0: 1 BL nodes, 0 FU nodes | 1 | 0 | 0 | False |
| 19f3cd308f | train | FU region 0: 3 BL nodes, 0 FU nodes | 3 | 0 | 0 | False |
| 5878a7ab84 | train | FU region 0: 2 BL nodes, 0 FU nodes | 2 | 0 | 0 | False |
| 8e98d81f82 | train | FU region 0: 0 BL nodes, 3 FU nodes | 11 | 3 | 0 | False |
| a3c65c2974 | train | FU region 0: 1 BL nodes, 0 FU nodes | 1 | 0 | 0 | False |
| b73ce398c3 | train | FU region 0: 3 BL nodes, 0 FU nodes | 3 | 0 | 0 | False |
| d296c101da | train | FU region 0: 1 BL nodes, 0 FU nodes | 1 | 0 | 0 | False |
| d947bf06a8 | train | FU region 0: 13 BL nodes, 0 FU nodes | 13 | 0 | 0 | False |
| e56954b4f6 | train | FU region 0: 1 BL nodes, 0 FU nodes | 1 | 0 | 0 | False |
| c6f057b865 | val | FU region 0: 0 BL nodes, 6 FU nodes; FU region 1: 1 BL nodes, 0 FU nodes | 0 | 6 | 0 | False |
| fc22130974 | val | FU region 0: 2 BL nodes, 0 FU nodes | 2 | 0 | 0 | False |
| 07e1cd7dca | test | FU region 0: 2 BL nodes, 0 FU nodes | 2 | 0 | 0 | True |
| 6883966fd8 | test | FU region 0: 3 BL nodes, 0 FU nodes | 3 | 0 | 0 | True |
| cfa0860e83 | test | FU region 0: 9 BL nodes, 0 FU nodes | 9 | 0 | 0 | True |
