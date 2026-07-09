# Graph Report - nanoUNet  (2026-07-06)

## Corpus Check
- 118 files · ~54,634 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1054 nodes · 1964 edges · 111 communities (53 shown, 58 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 60 edges (avg confidence: 0.6)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `54380904`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Blosc2 Data & Memory|Blosc2 Data & Memory]]
- [[_COMMUNITY_Preprocessing & Planning|Preprocessing & Planning]]
- [[_COMMUNITY_Training & Pretrain CLI|Training & Pretrain CLI]]
- [[_COMMUNITY_Inference & ROI Slices|Inference & ROI Slices]]
- [[_COMMUNITY_Config & Plans System|Config & Plans System]]
- [[_COMMUNITY_Longitudinal Registration|Longitudinal Registration]]
- [[_COMMUNITY_Loss Functions|Loss Functions]]
- [[_COMMUNITY_MAE Pretraining|MAE Pretraining]]
- [[_COMMUNITY_Resampling & Geometry|Resampling & Geometry]]
- [[_COMMUNITY_Plans Config3d|Plans Config3d]]
- [[_COMMUNITY_Labels & Metadata|Labels & Metadata]]
- [[_COMMUNITY_Cropping & Blosc2 IO|Cropping & Blosc2 IO]]
- [[_COMMUNITY_Predict CLI|Predict CLI]]
- [[_COMMUNITY_Longi DWB Model|Longi DWB Model]]
- [[_COMMUNITY_Normalization|Normalization]]
- [[_COMMUNITY_Slurm Finetune Registered|Slurm Finetune Registered]]
- [[_COMMUNITY_Slurm Preprocess Registered|Slurm Preprocess Registered]]
- [[_COMMUNITY_Slurm Longi Finetune|Slurm Longi Finetune]]
- [[_COMMUNITY_Slurm Sup 999 Overlap|Slurm Sup 999 Overlap]]
- [[_COMMUNITY_Slurm Sup Scratch XL|Slurm Sup Scratch XL]]
- [[_COMMUNITY_Geometry Utils|Geometry Utils]]
- [[_COMMUNITY_Plans & Philosophy Docs|Plans & Philosophy Docs]]
- [[_COMMUNITY_Longi Inference Pipeline|Longi Inference Pipeline]]
- [[_COMMUNITY_MAE Config & Runtime|MAE Config & Runtime]]
- [[_COMMUNITY_Data Augmentation|Data Augmentation]]
- [[_COMMUNITY_Slurm Preprocess Merge|Slurm Preprocess Merge]]
- [[_COMMUNITY_Slurm MAE Train|Slurm MAE Train]]
- [[_COMMUNITY_Inference Optimization|Inference Optimization]]
- [[_COMMUNITY_DWB Design Concepts|DWB Design Concepts]]
- [[_COMMUNITY_Loss & Patch Docs|Loss & Patch Docs]]
- [[_COMMUNITY_Checkpoint 412 Analysis|Checkpoint 412 Analysis]]
- [[_COMMUNITY_Dataset Utilities|Dataset Utilities]]
- [[_COMMUNITY_Checkpoint Tmpfs Leak|Checkpoint Tmpfs Leak]]
- [[_COMMUNITY_ResEnc & CNN-MAE|ResEnc & CNN-MAE]]
- [[_COMMUNITY_Mem Diag & Dataloader|Mem Diag & Dataloader]]
- [[_COMMUNITY_Longi Ablation Design|Longi Ablation Design]]
- [[_COMMUNITY_Patch Size Presets|Patch Size Presets]]
- [[_COMMUNITY_Package Root|Package Root]]
- [[_COMMUNITY_Pretrain Package|Pretrain Package]]
- [[_COMMUNITY_Cgroup Scope Diagnostics|Cgroup Scope Diagnostics]]
- [[_COMMUNITY_Multiprocessing Sharing|Multiprocessing Sharing]]
- [[_COMMUNITY_Build Plan Doc|Build Plan Doc]]
- [[_COMMUNITY_LR Schedule|LR Schedule]]
- [[_COMMUNITY_Standalone Port Plan|Standalone Port Plan]]
- [[_COMMUNITY_Longi Inference Doc|Longi Inference Doc]]
- [[_COMMUNITY_Landmark Align Plan|Landmark Align Plan]]
- [[_COMMUNITY_Registration Refine Plan|Registration Refine Plan]]
- [[_COMMUNITY_DWB Design Doc|DWB Design Doc]]
- [[_COMMUNITY_SOL Pressure Doc|SOL Pressure Doc]]
- [[_COMMUNITY_CLI Package Init|CLI Package Init]]
- [[_COMMUNITY_Data Package Init|Data Package Init]]
- [[_COMMUNITY_Infer Package Init|Infer Package Init]]
- [[_COMMUNITY_Model Package Init|Model Package Init]]
- [[_COMMUNITY_Plan Package Init|Plan Package Init]]
- [[_COMMUNITY_Register Package Init|Register Package Init]]
- [[_COMMUNITY_Pretrain Package Init|Pretrain Package Init]]
- [[_COMMUNITY_Root Package|Root Package]]
- [[_COMMUNITY_nanounet_standalone_port_4edf627b.plan|nanounet_standalone_port_4edf627b.plan.md]]
- [[_COMMUNITY_nanoUNet MAE pretraining|nanoUNet MAE pretraining]]
- [[_COMMUNITY_train.py|train.py]]
- [[_COMMUNITY_Host RAM  cgroup OOM (MAE & supervised training)|Host RAM / cgroup OOM (MAE & supervised training)]]
- [[_COMMUNITY_preprocessed_dir|preprocessed_dir]]
- [[_COMMUNITY_lightning_module.py|lightning_module.py]]
- [[_COMMUNITY_planner.py|planner.py]]
- [[_COMMUNITY_nanochat-style the nanoUNet way|nanochat-style: the nanoUNet way]]
- [[_COMMUNITY_Fix MAE  training cgroup OOM (Dataset999)|Fix MAE / training cgroup OOM (Dataset999)]]
- [[_COMMUNITY_common.py|common.py]]
- [[_COMMUNITY_config.py|config.py]]
- [[_COMMUNITY_NanoMAELM|NanoMAELM]]
- [[_COMMUNITY_slurm_nanounet_preprocess_d113_registered.sh|slurm_nanounet_preprocess_d113_registered.sh]]
- [[_COMMUNITY_io.py|io.py]]
- [[_COMMUNITY_planner_resenc.py|planner_resenc.py]]
- [[_COMMUNITY_Fix MAE cgroup OOM at the source|Fix MAE cgroup OOM at the source]]
- [[_COMMUNITY_slurm_nanounet_register_longi.sh|slurm_nanounet_register_longi.sh]]
- [[_COMMUNITY_.__init__|.__init__]]
- [[_COMMUNITY_MAE default num_workers=0|MAE default num_workers=0]]
- [[_COMMUNITY_Blosc2 page cache cgroup_file growth|Blosc2 page cache cgroup_file growth]]
- [[_COMMUNITY_longi_inference|longi_inference.md]]
- [[_COMMUNITY_posix_fadvise DONTNEED page cache eviction|posix_fadvise DONTNEED page cache eviction]]
- [[_COMMUNITY_Blosc2 lazy ROI patch reads|Blosc2 lazy ROI patch reads]]
- [[_COMMUNITY_border_expand BFS hull-shell inference|border_expand BFS hull-shell inference]]
- [[_COMMUNITY_4-mode patch sampling (posspurno_promptneg)|4-mode patch sampling (pos/spur/no_prompt/neg)]]
- [[_COMMUNITY_nanochat-style coding philosophy|nanochat-style coding philosophy]]
- [[_COMMUNITY_bottleneck-grid random masking 0.75|bottleneck-grid random masking 0.75]]
- [[_COMMUNITY_CNN-MAE self-supervised pretraining|CNN-MAE self-supervised pretraining]]
- [[_COMMUNITY_encoder-only MAE weight transfer|encoder-only MAE weight transfer]]
- [[_COMMUNITY_MAE fine-tune LR 1e-3 default recommendation|MAE fine-tune LR 1e-3 default recommendation]]
- [[_COMMUNITY_load_mae_encoder stem handling|load_mae_encoder stem handling]]
- [[_COMMUNITY_nanoUNet MAE pretraining plan|nanoUNet MAE pretraining plan]]
- [[_COMMUNITY_five-phase nnUNet port delivery|five-phase nnUNet port delivery]]
- [[_COMMUNITY_zero nnunetv2 runtime imports goal|zero nnunetv2 runtime imports goal]]
- [[_COMMUNITY_checkpoint staging tmpfs cgroup_shmem leak|checkpoint staging tmpfs cgroup_shmem leak]]
- [[_COMMUNITY_nanounetruntime.py set_safe_tmpdir|nanounet/runtime.py set_safe_tmpdir]]
- [[_COMMUNITY_--baseline-image co-located BL stream|--baseline-image co-located BL stream]]
- [[_COMMUNITY_--baseline-points pre-propagation BL centroids|--baseline-points pre-propagation BL centroids]]
- [[_COMMUNITY_Bouteille Learning to Look Closer ISBI 2026|Bouteille Learning to Look Closer ISBI 2026]]
- [[_COMMUNITY_DC+CE Dice cross-entropy default loss|DC+CE Dice cross-entropy default loss]]
- [[_COMMUNITY_nanoUNet losses documentation|nanoUNet losses documentation]]
- [[_COMMUNITY_patch size universal lesion segmentation doc|patch size universal lesion segmentation doc]]
- [[_COMMUNITY_inferborder_expand.py BFS expansion|infer/border_expand.py BFS expansion]]
- [[_COMMUNITY_dl-bucket and persistent-workers DataLoader prefs|dl-bucket and persistent-workers DataLoader prefs]]
- [[_COMMUNITY_nanounet_predict CLI|nanounet_predict CLI]]
- [[_COMMUNITY_nanounet_train CLI|nanounet_train CLI]]
- [[_COMMUNITY_Wald Revisiting MAE 3D Medseg CVPR 2025|Wald Revisiting MAE 3D Medseg CVPR 2025]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY_NanoMAELM|NanoMAELM]]
- [[_COMMUNITY_README|README.md]]

## God Nodes (most connected - your core abstractions)
1. `Plans` - 43 edges
2. `Config3d` - 36 edges
3. `Labels` - 35 edges
4. `cprint()` - 34 edges
5. `preprocessed_dir()` - 24 edges
6. `raw_dir()` - 21 edges
7. `nano_header()` - 20 edges
8. `convert_id_to_dataset_name()` - 20 edges
9. `build_pretrain_dataloaders()` - 20 edges
10. `main()` - 19 edges

## Surprising Connections (you probably didn't know these)
- `body_mask elastix fixed/moving masks` --conceptually_related_to--> `register()`  [EXTRACTED]
  docs/longi_registration_refine_plan.md → nanounet/register/elastix.py
- `checkpoint epoch 412 +1.3 per-lesion DSC win` --semantically_similar_to--> `best-epoch=412 checkpoint recommendation`  [INFERRED] [semantically similar]
  docs/session_2026-06-25_inference_optimization.md → README.md
- `landmark_align rigid FU<-BL transform` --conceptually_related_to--> `register()`  [EXTRACTED]
  docs/longi_registration_landmark_align_plan.md → nanounet/register/elastix.py
- `Host RAM cgroup OOM documentation` --references--> `MAE cgroup OOM fix plan`  [INFERRED]
  docs/cgroup_memory.md → .cursor/plans/fix_oom.md
- `MAE default num_workers=0` --conceptually_related_to--> `residual ~98 MB/epoch worker IPC shmem`  [INFERRED]
  .cursor/plans/fix_oom.md → docs/cgroup_memory.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **MAE cgroup OOM fix components** — _cursor_plans_fix_oom_posix_fadvise_dontneed, _cursor_plans_fix_oom_2_set_safe_tmpdir, _cursor_plans_fix_oom_mae_num_workers_zero, docs_cgroup_memory_nanounet_runtime, docs_cgroup_memory_mem_diag, nanounet_data_blosc2_dataset_blosc2folder [INFERRED 0.85]
- **Longitudinal DWB finetune end-to-end pipeline** — nanounet_cli_register_longi_register_longi, nanounet_cli_longi_build_longi_build, docs_longitudinal_dwb_design_build_patch_longi, docs_longitudinal_dwb_design_longiresencunet, nanounet_model_dwb_dwb, docs_longi_inference_baseline_image_flag [EXTRACTED 1.00]
- **BL to FU registration upgrade chain** — docs_longi_registration_refine_plan_geometric_center_init, docs_longi_registration_landmark_align_plan_landmark_align, docs_longi_registration_refine_plan_body_mask_metric, docs_longi_registration_refine_plan_refine_clicks, nanounet_register_elastix_register [INFERRED 0.85]

## Communities (111 total, 58 thin omitted)

### Community 0 - "Blosc2 Data & Memory"
Cohesion: 0.06
Nodes (34): 10.1 `docs/steps/predict.md`, 10.2 `docs/steps/longi.md` — rewrite the "Two-stream inference" section, 10. Docs (D2/D3/D4 — update in the SAME change), 11. Review checklist (run before done — SKILL §"Review checklist"), 1. Goal, 2.1 Stream/channel layout (unchanged model contract), 2.2 Null baseline (no baseline provided, or prompts disabled), 2. Why the current longi inference is wrong, and the fix (+26 more)

### Community 1 - "Preprocessing & Planning"
Cohesion: 0.08
Nodes (65): main(), Offline build of per-centroid sampling weights for d013 hard-type oversampling., main(), Build a 2-channel raw longi dataset from register_longi output.  Each FU case be, main(), Map warped BL clicks (register xyz, FU frame) into preprocessed voxels; write <c, main(), Fingerprint → plan (ResEnc) → preprocess 3d_fullres. (+57 more)

### Community 2 - "Training & Pretrain CLI"
Cohesion: 0.05
Nodes (65): ArgumentParser, IterableDataset, main(), Pretrain NanoUNet backbone with CNN-MAE (Lightning)., main(), build_train_parser(), Argparse + validation for nanounet_train., train_config_rows() (+57 more)

### Community 3 - "Inference & ROI Slices"
Cohesion: 0.06
Nodes (61): plan_border_expansion_centers_from_fg(), plan_border_expansion_centers_from_logits(), ndarray, slice, Tensor, BFS neighbors from hull-shell contact of foreground at patch faces (cc3d shell l, compute_gaussian(), gaussian_tile() (+53 more)

### Community 4 - "Config & Plans System"
Cohesion: 0.07
Nodes (49): Event, ClickModeConfig, InferenceConfig, LargeLesionConfig, load_config(), _load_inf(), _load_large(), _load_prompt() (+41 more)

### Community 5 - "Longitudinal Registration"
Cohesion: 0.07
Nodes (52): disjoint DICOM frame registration case 45fbc6d3e0_00, landmark_align rigid FU<-BL transform, MIN_SPREAD_MM collinear click guard, body_mask elastix fixed/moving masks, elastix GeometricalCenter transform init, main(), _process(), _quiet_stderr() (+44 more)

### Community 6 - "Loss Functions"
Cohesion: 0.11
Nodes (21): CC_DC_and_CE_loss, _cc_voronoi(), ndarray, Tensor, CC-DiceCE: global Dice+CE plus per-connected-component Dice+CE over GT Voronoi c, binary (Z,Y,X) → (cc_labels, voronoi_id, n_fg_cc). vor==0 if n==0., L = DC+CE (global) + λ · mean_{s,c} (DiceCE on Voronoi cell of CC c in sample s), get_tp_fp_fn_tn() (+13 more)

### Community 7 - "MAE Pretraining"
Cohesion: 0.15
Nodes (12): 1. Code style (distilled from Karpathy's nanochat), 2. UX: rich CLI, always, 3. Errors that teach, 4. GPU efficiency: compute is the bottleneck, always, 5. Documentation: small, structured, current, Hard rules, How nanoUNet adopts this (deviations, all deliberate), Naming (+4 more)

### Community 8 - "Resampling & Geometry"
Cohesion: 0.29
Nodes (9): estimate_conv_feature_map_size(), ndarray, ResidualEncoder 3D ``3d_fullres`` block: VRAM loop, topology, batch size (nnU-Ne, resenc_3d_fullres_plan(), ResEncPlannerPreset, get_pool_and_conv_props(), _pad_shape(), Pool strides + conv kernel sizes + patch pad so sizes divide by ``2**num_pool`` (+1 more)

### Community 9 - "Plans Config3d"
Cohesion: 0.11
Nodes (4): _bind_resample(), Config3d, _migrate_old_architecture(), One nnU-Net configuration block after `inherits_from` merge.

### Community 10 - "Labels & Metadata"
Cohesion: 0.12
Nodes (10): convert_logits_to_seg_shape(), export_prediction_from_logits(), ndarray, Tensor, Logits in preprocessed space → resample, uncrop, untranspose, SimpleITK seg writ, _collect_labels(), filter_background(), Labels (+2 more)

### Community 11 - "Cropping & Blosc2 IO"
Cohesion: 0.10
Nodes (15): _LRScheduler, pooled_fg_dice(), nnU-Net global pseudo-Dice: pool per-class fg tp/fp/fn over the whole val buffer, build_loss(), Module, PolyLRScheduler, Polynomial LR + stretched-tail variant (nnU-Net)., StretchedTailPolyLRScheduler (+7 more)

### Community 12 - "Predict CLI"
Cohesion: 0.12
Nodes (20): main(), Dataset / single-case prompt-driven inference with CPU prefetch., config_table(), Any, Render resolved config as a rich Table: (argument, value, source: cli/config/def, _assert_bl_geometry(), baseline_resolver(), check_baseline_files() (+12 more)

### Community 13 - "Longi DWB Model"
Cohesion: 0.35
Nodes (7): ndarray, SimpleITK reader/writer: nnU-Net channel stacking, ``sitk_stuff`` in properties,, read_images(), read_seg(), _same_shape(), SimpleITKIO, write_seg()

### Community 14 - "Normalization"
Cohesion: 0.07
Nodes (35): ndarray, crop_to_nonzero(), _nonzero_mask(), ndarray, Nonzero bounding-box crop; seg voxels outside mask set to ``nonzero_label`` (def, CTNormalization, NoNormalization, normalization_class_from_plan_name() (+27 more)

### Community 16 - "Slurm Finetune Registered"
Cohesion: 0.14
Nodes (13): MKL_NUM_THREADS, NANOUNET_PREPROCESSED, NANOUNET_RAW, NANOUNET_RESULTS, NANOUNET_TMPDIR, nnUNet_preprocessed, nnUNet_raw, nnUNet_results (+5 more)

### Community 17 - "Slurm Preprocess Registered"
Cohesion: 0.08
Nodes (24): 0.5. Coding philosophy and style (read this first), 0. Goals (in scope), 10. CLIs and entry points, 11. Reproducibility and resume, 12. Performance discipline (no GPU starvation), 13. Output folder layout, 14. Validation and removal-after-test plan, 15. Optional follow-ups (NOT in v1) (+16 more)

### Community 18 - "Slurm Longi Finetune"
Cohesion: 0.18
Nodes (10): MKL_NUM_THREADS, NANOUNET_PREPROCESSED, NANOUNET_RAW, NANOUNET_RESULTS, NANOUNET_TMPDIR, NUMEXPR_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS (+2 more)

### Community 19 - "Slurm Sup 999 Overlap"
Cohesion: 0.18
Nodes (10): MKL_NUM_THREADS, NANOUNET_PREPROCESSED, NANOUNET_RAW, NANOUNET_RESULTS, NANOUNET_TMPDIR, NUMEXPR_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS (+2 more)

### Community 20 - "Slurm Sup Scratch XL"
Cohesion: 0.18
Nodes (10): MKL_NUM_THREADS, NANOUNET_PREPROCESSED, NANOUNET_RAW, NANOUNET_RESULTS, NANOUNET_TMPDIR, NUMEXPR_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS (+2 more)

### Community 21 - "Geometry Utils"
Cohesion: 0.29
Nodes (9): insert_crop(), nonzero_slices_3d(), ndarray, slice, Tensor, Nonzero crop, transpose, resample delegate, insert crop (acvl_utils)., resample_tensor(), transpose_backward_np() (+1 more)

### Community 22 - "Plans & Philosophy Docs"
Cohesion: 0.67
Nodes (3): MAE cgroup OOM fix plan v2 (TMPDIR), MAE cgroup OOM fix plan, Host RAM cgroup OOM documentation

### Community 23 - "Longi Inference Pipeline"
Cohesion: 0.33
Nodes (6): refine_clicks per-lesion VOI instance optimization, build_patch_longi 6-channel patch construction, stamping baseline collage rejected, warp-based two-stream finetune pipeline, nanounet.cli.longi_build 2-channel dataset, nanounet.cli.register_longi pipeline

### Community 25 - "Data Augmentation"
Cohesion: 0.40
Nodes (5): Documentation, Environment, Install, nanoUNet, Smoke test

### Community 26 - "Slurm Preprocess Merge"
Cohesion: 0.33
Nodes (5): NANOUNET_PREPROCESSED, NANOUNET_RAW, NANOUNET_RESULTS, PIP_CACHE_DIR, slurm_nanounet_preprocess_merge_999.sh script

### Community 27 - "Slurm MAE Train"
Cohesion: 0.33
Nodes (5): NANOUNET_PREPROCESSED, NANOUNET_RAW, NANOUNET_RESULTS, PIP_CACHE_DIR, slurm_nanounet_train_mae_999.sh script

### Community 29 - "DWB Design Concepts"
Cohesion: 0.50
Nodes (4): Difference Weighting Block (DWB), LongiResEncUNet two-stream network, LongiSeg longitudinal segmentation SOTA, nanounet/model/dwb.py DWB block

### Community 30 - "Loss & Patch Docs"
Cohesion: 0.67
Nodes (3): CC-DiceCE connected-component Voronoi loss, dual-scale lesion patch playbook, Primus patch-size sensitivity arXiv:2503.01835

### Community 31 - "Checkpoint 412 Analysis"
Cohesion: 0.67
Nodes (3): checkpoint epoch 412 +1.3 per-lesion DSC win, train/inference distribution match audit, best-epoch=412 checkpoint recommendation

### Community 32 - "Dataset Utilities"
Cohesion: 0.20
Nodes (10): `click_modes` constraint, Common errors, Example, `inference`, `prompt`, ROI / prompt configuration, `sampling`, Top-level sections (+2 more)

### Community 59 - "nanounet_standalone_port_4edf627b.plan.md"
Cohesion: 0.10
Nodes (20): Coding philosophy and style — non-negotiable, Constraints, Goal, Hard rules (apply to every file in `nanounet/`), Key snippets to mirror, Module layout after the port, Naming, Phase 1 — Plans/labels + network + losses (in-place training works without nnunetv2) (+12 more)

### Community 60 - "nanoUNet MAE pretraining"
Cohesion: 0.10
Nodes (19): 0. What we are building (and what we explicitly are not), 1. Evidence backing the default recipe, 2. Architecture choice for MAE (the only non-obvious design call), 3.1 [`nanounet/pretrain/masking.py`](nanoUNet/nanounet/pretrain/masking.py), 3.2 [`nanounet/pretrain/dataset.py`](nanoUNet/nanounet/pretrain/dataset.py), 3.3 [`nanounet/pretrain/module.py`](nanoUNet/nanounet/pretrain/module.py), 3.4 [`nanounet/model/mae_transfer.py`](nanoUNet/nanounet/model/mae_transfer.py), 3.5 [`nanounet/cli/pretrain.py`](nanoUNet/nanounet/cli/pretrain.py) (+11 more)

### Community 61 - "train.py"
Cohesion: 0.25
Nodes (8): Longitudinal workflow, `nanounet_longi_build`, `nanounet_longi_clicks`, `nanounet_register_longi`, `nanounet_repair_longi_fu`, Pipeline, Training finetune, Two-stream inference

### Community 62 - "Host RAM / cgroup OOM (MAE & supervised training)"
Cohesion: 0.12
Nodes (16): 1. Checkpoint temp files on RAM-backed `/tmp` (main OOM), 2. Page cache from Blosc2 reads (secondary), 3. Misleading metrics on the interactive node, Fixes in code, Home disk quota, Host RAM / cgroup OOM (MAE & supervised training), Monitoring (historical: `--mem-diag`), Purge safety (Docker / shared node) (+8 more)

### Community 63 - "preprocessed_dir"
Cohesion: 0.29
Nodes (7): `batch_dice` and CC-DiceCE, Caveats, CC-DiceCE (opt-in), Cost / deps, Losses, Throughput warning, Why not Blob Loss?

### Community 64 - "lightning_module.py"
Cohesion: 0.29
Nodes (7): Arguments (planning-related), Command, Common errors, Further reading, Inputs / outputs, Planner presets, Planning knobs

### Community 65 - "planner.py"
Cohesion: 0.29
Nodes (7): Arguments, Command, Common errors, Host RAM / cgroup OOM, Inputs / outputs, Loss throughput, Supervised training

### Community 66 - "nanochat-style: the nanoUNet way"
Cohesion: 0.33
Nodes (6): BasicTransform, ndarray, Spatial + intensity augment chains (nnUNetTrainer.get_*_transforms port, 3D only, train_transforms(), val_transforms(), RandomScalar

### Community 67 - "Fix MAE / training cgroup OOM (Dataset999)"
Cohesion: 0.17
Nodes (11): 1. Move tempfile root off tmpfs (primary OOM fix), 2. Detect & prune the tmpfs checkpoint leak, 3. Honest cgroup diagnostics, 4. Prove Fix A is live, 5. One-shot cleanup + Slurm validation, Diagnosis confirmed from the live host, Files touched, Fix MAE / training cgroup OOM (Dataset999) (+3 more)

### Community 68 - "common.py"
Cohesion: 0.33
Nodes (6): Arguments, Checkpoint selection, Command, Common errors, Inputs / outputs, Predict

### Community 69 - "config.py"
Cohesion: 0.53
Nodes (3): ndarray, Tensor, _softmax_class_dim()

### Community 70 - "NanoMAELM"
Cohesion: 0.15
Nodes (23): _cgroup_dir(), cgroup_scope(), Path, Host resource readers: cgroup scope, mount fs-type (used by TMPDIR selection + p, tmp_fs_type(), Runtime resource plumbing: cgroup scope, tmpfs detection, orphan temp-file purge, _is_pytorch_zip_stage(), _purge_ckpt_stage_files() (+15 more)

### Community 71 - "slurm_nanounet_preprocess_d113_registered.sh"
Cohesion: 0.17
Nodes (11): MKL_NUM_THREADS, NANOUNET_PREPROCESSED, NANOUNET_RAW, NANOUNET_RESULTS, nnUNet_preprocessed, nnUNet_raw, nnUNet_results, NUMEXPR_NUM_THREADS (+3 more)

### Community 72 - "io.py"
Cohesion: 0.20
Nodes (12): load_net_from_ckpt(), device, Checkpoint load: strip Lightning prefix, build net, pick ckpt file., _strip_pl_state(), _build_class(), build_net(), build_net_longi(), Instantiate ResidualEncoderUNet (optional extra input channels for prompts) from (+4 more)

### Community 73 - "planner_resenc.py"
Cohesion: 0.40
Nodes (5): Arguments, Command, Common errors, Inputs / outputs, Preprocess

### Community 74 - "Fix MAE cgroup OOM at the source"
Cohesion: 0.22
Nodes (8): Fix A: Page-cache eviction (the only sound way to stop `cgroup_file`), Fix B: Eliminate DataLoader IPC shmem leakage, Fix C: Diagnostics to prove the fix, Fix MAE cgroup OOM at the source, Out of scope, Root cause (confirmed), Sequence diagram of the read+evict flow, Validation protocol (success criteria)

### Community 75 - "slurm_nanounet_register_longi.sh"
Cohesion: 0.29
Nodes (7): complete(), MKL_NUM_THREADS, NANOUNET_REG_THREADS, NUMEXPR_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, slurm_nanounet_register_longi.sh script

### Community 76 - ".__init__"
Cohesion: 0.40
Nodes (5): Arguments, Command, Common errors, Inputs / outputs, MAE pretrain

### Community 79 - "longi_inference.md"
Cohesion: 0.50
Nodes (4): Documentation map, nanoUNet documentation, Pipeline overview, Quickstart

### Community 109 - "NanoMAELM"
Cohesion: 0.16
Nodes (7): BaseException, bottleneck_mask(), device, Tensor, Bottleneck-grid random masks upsampled to voxel resolution (Spark3D-style)., NanoMAELM, Tensor

## Knowledge Gaps
- **297 isolated node(s):** `nanounet`, `slurm_nanounet_finetune_d013_longi.sh script`, `PIP_CACHE_DIR`, `NANOUNET_RAW`, `NANOUNET_RESULTS` (+292 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **58 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `cprint()` connect `Preprocessing & Planning` to `Training & Pretrain CLI`, `Longitudinal Registration`, `NanoMAELM`, `Predict CLI`, `Longi DWB Model`, `Normalization`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Why does `Plans` connect `Training & Pretrain CLI` to `Preprocessing & Planning`, `io.py`, `Labels & Metadata`, `Cropping & Blosc2 IO`, `Predict CLI`, `NanoMAELM`, `Normalization`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Why does `Config3d` connect `Plans Config3d` to `Training & Pretrain CLI`, `Loss Functions`, `io.py`, `Labels & Metadata`, `Cropping & Blosc2 IO`, `Normalization`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `Plans` (e.g. with `Labels` and `PretrainPatchIterable`) actually correct?**
  _`Plans` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `Config3d` (e.g. with `DC_and_CE_loss` and `DeepSupervisionWrapper`) actually correct?**
  _`Config3d` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `Labels` (e.g. with `DC_and_CE_loss` and `DeepSupervisionWrapper`) actually correct?**
  _`Labels` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `nanoUNet: prompt-aware 3D ResEnc U-Net (Lightning + bundled preprocess/infer).`, `CLI entry points: preprocess, train, predict, pretrain, longi registration.`, `Offline build of per-centroid sampling weights for d013 hard-type oversampling.` to the rest of the system?**
  _422 weakly-connected nodes found - possible documentation gaps or missing edges._