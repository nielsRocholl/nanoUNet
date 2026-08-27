"""Build SegTrackCase list from argparse: folder vs single, optional BL instance mask."""
from __future__ import annotations

from pathlib import Path

from nanounet.infer.predict_io import patient_ids_from_csv
from nanounet.infer.segtrack_case import SegTrackCase, pair_folder, stem_pid_region


def collect_cases(args) -> tuple[list[SegTrackCase], bool, list[tuple[str, str]], tuple[Path | None, str]]:
    folder = bool(args.bl_dir) or bool(args.fu_dir)
    single = any((args.bl_img, args.bl_clicks, args.bl_mask, args.fu_img, args.fu_clicks))
    if folder == single:
        raise SystemExit(
            "Need either folder mode (--bl-dir --fu-dir) or single case (--bl-img --bl-clicks --fu-img --fu-clicks).\n"
            "Expected one mode, not both or neither.\n"
            "Fix: see docs/steps/track.md"
        )
    if args.bl_mask and args.bl_mask_dir:
        raise SystemExit(
            "--bl-mask and --bl-mask-dir were both set.\n"
            "Expected one instance-mask input, or neither.\n"
            "Fix: --bl-mask <file> in single mode, or --bl-mask-dir <folder> in folder mode  (see docs/steps/track.md)"
        )
    if args.bl_mask and folder:
        raise SystemExit(
            "--bl-mask is single mode.\n"
            "Expected --bl-mask-dir with --bl-dir --fu-dir.\n"
            "Fix: --bl-mask-dir /nnunet_data/Longitudinal-CT/targetsTrBL  (see docs/steps/track.md)"
        )
    if args.bl_mask_dir and not folder:
        raise SystemExit(
            "--bl-mask-dir is folder mode.\n"
            "Expected --bl-mask with --bl-img --fu-img --fu-clicks.\n"
            "Fix: --bl-mask /nnunet_data/Longitudinal-CT/targetsTrBL/<stem>.nii.gz  (see docs/steps/track.md)"
        )
    if folder:
        cases, skipped, meta = _folder(args)
        return cases, False, skipped, meta
    cases, meta = _single(args)
    return cases, True, [], meta


def _folder(args) -> tuple[list[SegTrackCase], list[tuple[str, str]], tuple[Path | None, str]]:
    if not (args.bl_dir and args.fu_dir):
        raise SystemExit("--bl-dir requires --fu-dir.\nExpected both folders.\nFix: see docs/steps/track.md")
    if args.meta:
        raise SystemExit("--meta is for single mode.\nExpected --meta-dir in folder mode.\nFix: --meta-dir /path/to/meta")
    pids = patient_ids_from_csv(args.patients_csv) if args.patients_csv else None
    cases, skipped = pair_folder(
        Path(args.bl_dir), Path(args.fu_dir),
        bl_mask_dir=Path(args.bl_mask_dir) if args.bl_mask_dir else None, pids=pids,
    )
    inferred = Path(args.bl_dir).resolve().parent / "meta"
    if args.meta_dir:
        md, src = Path(args.meta_dir), "cli"
    elif inferred.is_dir():
        md, src = inferred, "inferred"
    else:
        md, src = None, "none"
    if md is not None:
        keep = []
        for c in cases:
            pid, _ = stem_pid_region(c.stem)
            p = md / f"{pid}.csv"
            if not p.is_file():
                skipped.append((c.stem, "no meta csv"))
                continue
            c.meta_csv = p
            c.types_csv = p
            keep.append(c)
        cases = keep
    return cases, skipped, (md, src)


def _single(args) -> tuple[list[SegTrackCase], tuple[Path | None, str]]:
    if args.meta_dir or args.patients_csv:
        raise SystemExit("--meta-dir / --patients-csv are folder mode.\nExpected --meta for one case.\nFix: see docs/steps/track.md")
    types = Path(args.meta) if args.meta else None
    if types is not None and not types.is_file():
        raise SystemExit(
            f"No types CSV at {types}.\nExpected lesion_id, lesion_type.\nFix: --meta <pid>.csv or omit it  (see docs/steps/track.md)"
        )
    if args.bl_mask:
        if args.bl_clicks:
            raise SystemExit(
                "--bl-clicks was set with --bl-mask.\n"
                "Expected the baseline instance mask to supply ids; BL clicks are not used.\n"
                "Fix: drop --bl-clicks  (see docs/steps/track.md)"
            )
        if not all((args.bl_img, args.bl_mask, args.fu_img, args.fu_clicks)):
            raise SystemExit(
                "Single mask mode needs --bl-img --bl-mask --fu-img --fu-clicks.\n"
                "Expected four paths (no --bl-clicks).\nFix: see docs/steps/track.md"
            )
        if not Path(args.bl_mask).is_file():
            raise SystemExit(
                f"No BL mask at {args.bl_mask}.\n"
                "Expected a native baseline instance NIfTI or .mha (voxel = lesion_id).\n"
                "Fix: --bl-mask /nnunet_data/Longitudinal-CT/targetsTrBL/<stem>.nii.gz  (see docs/steps/track.md)"
            )
        stem = Path(args.fu_img).name[:-7] if Path(args.fu_img).name.endswith(".nii.gz") else Path(args.fu_img).name
        cases = [SegTrackCase(stem, Path(args.bl_img), None, Path(args.fu_img), Path(args.fu_clicks), types, Path(args.bl_mask))]
    elif not all((args.bl_img, args.bl_clicks, args.fu_img, args.fu_clicks)):
        raise SystemExit(
            "Single mode needs --bl-img --bl-clicks --fu-img --fu-clicks.\n"
            "Expected four paths.\nFix: see docs/steps/track.md"
        )
    else:
        stem = Path(args.fu_img).name[:-7] if Path(args.fu_img).name.endswith(".nii.gz") else Path(args.fu_img).name
        cases = [SegTrackCase(stem, Path(args.bl_img), Path(args.bl_clicks), Path(args.fu_img), Path(args.fu_clicks), types)]
    c = cases[0]
    if types is not None:
        c.meta_csv = types
        c.types_csv = types
        return cases, (types.parent, "cli")
    inferred = Path(args.bl_img).resolve().parent.parent / "meta"
    pid, _ = stem_pid_region(c.stem)
    p = inferred / f"{pid}.csv"
    if inferred.is_dir() and p.is_file():
        c.meta_csv = p
        c.types_csv = p
        return cases, (inferred, "inferred")
    return cases, (None, "none")
