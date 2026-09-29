# nanoUNet monorepo

CT lesion research code, one project per top-level folder. Each folder is self-contained: code, `README.md`, `docs/`, `configs/`, `scripts/`.

| project | what | start here |
|---|---|---|
| [`nanounet/`](nanounet/) | promptable 3D lesion segmentation (ResEnc nnU-Net, Lightning) | [`nanounet/README.md`](nanounet/README.md) |
| [`lesionglue/`](lesionglue/) | LesionGlue: BL↔FU lesion matching with a dense PyG GNN | [`lesionglue/README.md`](lesionglue/README.md) |

```bash
pip install -e ".[lesionglue]"
```
