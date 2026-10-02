# nanoUNet

Minimal prompt-aware 3D ResEnc U-Net with PyTorch Lightning and optional MAE pretraining. Layout and style follow [nanochat](https://github.com/karpathy/nanochat): small modules, no framework sprawl. The U-Net preprocessing, training, and setup pipeline draws a lot of inspiration from [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).

## Install

```bash
python -m pip install -e .
```

## Environment

```bash
export NANOUNET_RAW="/path/to/NanoUNet_raw"
export NANOUNET_PREPROCESSED="/path/to/NanoUNet_preprocessed"
export NANOUNET_RESULTS="/path/to/NanoUNet_results"
# nnFoundationCNN weights cache (downloaded once by nanounet_preprocess; CC-BY-SA-4.0)
export NANOUNET_PRETRAINED="/path/to/NanoUNet_pretrained"

# Host-RAM / checkpoint staging (see nanounet/docs/dev-notes/cgroup_memory.md)
export NANOUNET_TMPDIR=/root/.cache/nanounet_tmp
```

Quote paths that contain spaces.

## Smoke test

```bash
python -c "import sys; import nanounet.cli.preprocess, nanounet.cli.train, nanounet.cli.predict; assert 'nnunetv2' not in sys.modules; print('ok')"
```



## Pretrained weights and license

`nanounet_preprocess` downloads the DKFZ nnFoundationCNN encoder (Hugging Face `MIC-DKFZ/nnFoundationCNN`,
license **CC-BY-SA-4.0**: share derived weights under the same license, with attribution) and `nanounet_train`
starts from it. Cite:

```bibtex
@misc{harsy2026nnfoundation3dfoundationmodels,
  title={nnFoundation: 3D Foundation Models for Radiology},
  author={Constantin Ulrich Harsy and Tassilo Wald and Karol Gotkowski and others and Fabian Isensee and Klaus H. Maier-Hein},
  year={2026}, eprint={2609.26924}, archivePrefix={arXiv}, primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.26924}
}
```

## Documentation


| Resource                       | Link                                                               |
| ------------------------------ | ------------------------------------------------------------------ |
| Pipeline overview & quickstart | [nanounet/docs/index.md](docs/index.md)                                     |
| Preprocess                     | [nanounet/docs/steps/preprocess.md](docs/steps/preprocess.md)               |
| Planning knobs                 | [nanounet/docs/steps/plan.md](docs/steps/plan.md)                           |
| MAE pretrain                   | [nanounet/docs/steps/pretrain.md](docs/steps/pretrain.md)                   |
| Supervised train               | [nanounet/docs/steps/train.md](docs/steps/train.md)                         |
| Inference                      | [nanounet/docs/steps/predict.md](docs/steps/predict.md)                     |
| Track (seg × track)            | [segtrack/README.md](../segtrack/README.md)                         |
| Fixed valset                   | [nanounet/docs/steps/valset.md](docs/steps/valset.md)                       |
| Lesion weights                 | [nanounet/docs/steps/lesion_weights.md](docs/steps/lesion_weights.md)       |
| Instance targets               | [nanounet/docs/reference/instance_targets.md](docs/reference/instance_targets.md) |
| Tracking ids                   | [segtrack/docs/track_ids.md](../segtrack/docs/track_ids.md)         |
| ROI / prompt config            | [nanounet/docs/reference/config.md](docs/reference/config.md)               |
| Patch size playbook            | [nanounet/docs/reference/patch_size.md](docs/reference/patch_size.md)       |
| Loss functions                 | [nanounet/docs/reference/losses.md](docs/reference/losses.md)               |
| Host RAM / cgroup OOM          | [nanounet/docs/dev-notes/cgroup_memory.md](docs/dev-notes/cgroup_memory.md) |


Entry points: `nanounet_preprocess`, `nanounet_train`, `nanounet_pretrain`, `nanounet_predict`, `segtrack_run`, `nanounet_build_splits`, `nanounet_build_valset`, `nanounet_lesion_weights` (see [pyproject.toml](../pyproject.toml)).