# exp00c segmentation evaluation manifest

cases: {'seen-cohort': 234, 'outside': 270, 'healthy': 172}; dropped by header: 99; manifest: /nanoUNet/.claude/worktrees/agent-a81a0c5a44cf057bf/experiments/exp00c_seg_eval_manifest/seg_eval_v1.json

| tier | source | cancer type | cases | patients | lesions |
|---|---|---|---|---|---|
| healthy | HealthyImages-noLesion:CHAOS | none | 40 | 40 | 0 |
| healthy | HealthyImages-noLesion:PANCREAS | none | 82 | 82 | 0 |
| healthy | HealthyImages-noLesion:train | none | 50 | 50 | 0 |
| outside | AbdomenAtlas-endometrial | endometrial | 30 | 30 | 42 |
| outside | AbdomenAtlas-esophagus | esophagus | 30 | 30 | 30 |
| outside | BoneTumorLungMetastasis | lung | 30 | 30 | 532 |
| outside | HCC-TACE-SEG | liver | 30 | 30 | 77 |
| outside | NSCLC-Radiogenomics | lung | 30 | 30 | 68 |
| outside | PANORAMA | pancreas | 30 | 30 | 30 |
| outside | ScienceDBLungTumorCT | lung | 30 | 30 | 39 |
| outside | SegRap25 | head-neck | 30 | 30 | 352 |
| outside | SpinalMyelomaCT | bone | 30 | 30 | 1469 |
| seen-cohort | Longitudinal-CT-holdout60 | melanoma | 132 | 60 | 1141 |
| seen-cohort | val:Adrenal-ACC-Ki67 | adrenal | 5 | 5 | 5 |
| seen-cohort | val:CLM | liver | 5 | 5 | 15 |
| seen-cohort | val:KiTS23 | kidney | 5 | 5 | 5 |
| seen-cohort | val:LIDC-IDRI | lung | 5 | 5 | 17 |
| seen-cohort | val:LNDb | lung | 5 | 5 | 17 |
| seen-cohort | val:LiTS | liver | 5 | 5 | 32 |
| seen-cohort | val:Longitudinal-CT | skin_melanoma | 5 | 5 | 49 |
| seen-cohort | val:MCT-LTDiag | liver | 5 | 5 | 10 |
| seen-cohort | val:MSD-Colon | colon | 5 | 5 | 5 |
| seen-cohort | val:MSD-Liver | liver | 5 | 5 | 18 |
| seen-cohort | val:MSD-Lung | lung | 5 | 5 | 9 |
| seen-cohort | val:MSD-Pancreas | pancreas | 5 | 5 | 5 |
| seen-cohort | val:MSWAL | abdomen_mixed | 5 | 5 | 5 |
| seen-cohort | val:Mediastinal-LN | lymph_node | 5 | 5 | 37 |
| seen-cohort | val:PanTS | pancreas | 5 | 5 | 6 |
| seen-cohort | val:RIDER-LungCT | lung | 5 | 5 | 5 |
| seen-cohort | val:RUMC-Bone | bone | 5 | 5 | 35 |
| seen-cohort | val:RUMC-Pancreas | pancreas | 2 | 2 | 3 |
| seen-cohort | val:WAW-TACE | liver | 5 | 5 | 8 |
| seen-cohort | val:WORC-CRLM | liver | 5 | 5 | 5 |
| seen-cohort | val:WORC-GIST | gist | 5 | 5 | 9 |

| source | candidates | dropped by header | patients alive | selected |
|---|---|---|---|---|
| SegRap25 | 120 | 0 | 120 | 30 |
| ScienceDBLungTumorCT | 1167 | 0 | 1167 | 30 |
| NSCLC-Radiogenomics | 88 | 0 | 88 | 30 |
| HCC-TACE-SEG | 74 | 0 | 74 | 30 |
| SpinalMyelomaCT | 72 | 0 | 67 | 30 |
| BoneTumorLungMetastasis | 61 | 0 | 61 | 30 |
| PANORAMA | 479 | 99 | 379 | 30 |
| AbdomenAtlas-esophagus | 154 | 0 | 154 | 30 |
| AbdomenAtlas-endometrial | 79 | 0 | 79 | 30 |
