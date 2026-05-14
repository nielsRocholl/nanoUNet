"""Preprocess one case, ROI predict, export segmentation."""

from __future__ import annotations

import argparse

from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nanounet.common import cprint, nano_header, sync_nnunet_env
from nanounet.infer.border_expand import DEFAULT_MAX_BORDER_EXPAND_EXTRA
from nanounet.infer.export import export_prediction_from_logits
from nanounet.infer.predictor import pick_checkpoint, predict_logits_preprocessed
from nanounet.plan.plans import Plans


def main() -> None:
    sync_nnunet_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", nargs="+", required=True)
    ap.add_argument("-o", "--output", required=True, help="output path (.nii.gz or base without suffix)")
    ap.add_argument("-m", "--model-dir", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--point-zyx", required=True, help="comma z,y,x (preprocessed unpadded voxel)")
    ap.add_argument("--seg", default=None)
    ap.add_argument("--no-prompt-encode", action="store_true")
    ap.add_argument("--border-expand", action="store_true")
    ap.add_argument("--max-border-extra", type=int, default=DEFAULT_MAX_BORDER_EXPAND_EXTRA)
    ap.add_argument("--disable-tta", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    nano_header("nanoUNet predict", color="blue")
    pt = tuple(int(x.strip()) for x in args.point_zyx.split(","))
    if len(pt) != 3:
        raise SystemExit("need z,y,x")
    ck = pick_checkpoint(args.model_dir, args.ckpt)
    logits, props = predict_logits_preprocessed(
        args.images,
        args.seg,
        args.model_dir,
        ck,
        [pt],
        encode_prompt=not args.no_prompt_encode,
        border_expand=args.border_expand,
        max_border_expand_extra=args.max_border_extra,
        disable_tta=args.disable_tta,
        device=args.device,
    )
    dj = load_json(join(args.model_dir, "dataset.json"))
    end = dj["file_ending"]
    out = args.output
    trunc = out[: -len(end)] if out.endswith(end) else out
    pl = Plans(join(args.model_dir, "plans.json"))
    cm = pl.get_configuration("3d_fullres")
    export_prediction_from_logits(logits, props, cm, pl, dj, trunc)
    cprint(f"[bold green]✓ wrote {trunc}{end}[/bold green]")


if __name__ == "__main__":
    main()
