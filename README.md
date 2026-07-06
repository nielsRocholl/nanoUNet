# nanoUNet

Minimal prompt-aware 3D ResEnc U-Net with PyTorch Lightning and optional MAE pretraining; optional longitudinal finetune uses a registered BL+FU dual-stream encoder with difference weighting at skips. Layout and style follow [nanochat](https://github.com/karpathy/nanochat): small modules, no framework sprawl. The U-Net preprocessing, training, and setup pipeline draws a lot of inspiration from [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).

## Install

```bash
python -m pip install -e .
```

## Environment

```bash
export NANOUNET_RAW="/path/to/NanoUNet_raw"
export NANOUNET_PREPROCESSED="/path/to/NanoUNet_preprocessed"
export NANOUNET_RESULTS="/path/to/NanoUNet_results"

# Host-RAM / checkpoint staging (see docs/dev-notes/cgroup_memory.md)
export NANOUNET_TMPDIR=/root/.cache/nanounet_tmp
export NANOUNET_ALLOW_ROOT_CGROUP=1   # only if using --mem-diag on interactive node
```

Quote paths that contain spaces.

## Smoke test

```bash
python -c "import sys; import nanounet.cli.preprocess, nanounet.cli.train, nanounet.cli.predict; assert 'nnunetv2' not in sys.modules; print('ok')"
```



## Documentation


| Resource                       | Link                                                               |
| ------------------------------ | ------------------------------------------------------------------ |
| Pipeline overview & quickstart | [docs/index.md](docs/index.md)                                     |
| Preprocess                     | [docs/steps/preprocess.md](docs/steps/preprocess.md)               |
| Planning knobs                 | [docs/steps/plan.md](docs/steps/plan.md)                           |
| MAE pretrain                   | [docs/steps/pretrain.md](docs/steps/pretrain.md)                   |
| Supervised train               | [docs/steps/train.md](docs/steps/train.md)                         |
| Inference                      | [docs/steps/predict.md](docs/steps/predict.md)                     |
| Longitudinal workflow          | [docs/steps/longi.md](docs/steps/longi.md)                         |
| ROI / prompt config            | [docs/reference/config.md](docs/reference/config.md)               |
| Patch size playbook            | [docs/reference/patch_size.md](docs/reference/patch_size.md)       |
| Loss functions                 | [docs/reference/losses.md](docs/reference/losses.md)               |
| Host RAM / cgroup OOM          | [docs/dev-notes/cgroup_memory.md](docs/dev-notes/cgroup_memory.md) |


Entry points: `nanounet_preprocess`, `nanounet_train`, `nanounet_pretrain`, `nanounet_predict` (see [pyproject.toml](pyproject.toml)).