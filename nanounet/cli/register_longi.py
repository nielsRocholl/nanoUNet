"""Warp BL/FU pairs into the FU frame (itk-elastix) and write a mini dataset + optional QC.

  # one pair
  python -m nanounet.cli.register_longi --data-root <DIR> --out <DIR> --pid 006f52e910 --idx 00 --qc
  # N random pairs (only cases that have both BL and FU)
  python -m nanounet.cli.register_longi --data-root <DIR> --out <DIR> --sample 5 --qc
  # every pair
  python -m nanounet.cli.register_longi --data-root <DIR> --out <DIR> --all --qc

Per case writes inputsTrBL/targetsTrBL (warped BL CT/mask + clicks in FU frame) plus copied FU + meta.
Cases that cannot register (missing BL, elastix failure, all lesions vanished) are skipped with a
reason and do not abort the batch; exit code is nonzero if any case skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
import random
import sys
import time

import itk
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from nanounet.common import _CONSOLE, cprint, nano_header
from nanounet.register.output import qc_png, write_dataset
from nanounet.register.resources import apply_threads, resolve_threads
from nanounet.register.warp_case import warp_case


@contextlib.contextmanager
def _quiet_stderr(verbose: bool):
    if verbose:
        yield
        return
    with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
        yield


def _select_cases(args) -> list[tuple[str, str]]:
    if args.pid and args.idx:
        return [(args.pid, args.idx)]
    fu = glob.glob(os.path.join(args.data_root, "inputsTrFU", "*.nii.gz"))
    cases = []
    for f in fu:
        stem = os.path.basename(f)[: -len(".nii.gz")]
        pid, idx = stem.rsplit("_", 1)
        if os.path.isfile(os.path.join(args.data_root, "inputsTrBL", f"{stem}.nii.gz")):
            cases.append((pid, idx))
    cases.sort()
    assert cases, f"no BL/FU pairs under {args.data_root}"
    if args.sample:
        random.seed(args.seed)
        cases = random.sample(cases, min(args.sample, len(cases)))
    return cases


def _process(args, pid: str, idx: str, threads: int, on_step) -> tuple[int, int, list[str]]:
    """Returns (n_clicks, warped_vox, out_paths). Raises on any case-level failure."""
    res = warp_case(
        args.data_root, pid, idx,
        body_mask_metric=not args.no_body_mask,
        refine=not args.no_refine,
        threads=threads,
        verbose=args.verbose,
        on_step=on_step,
    )
    on_step("write")
    paths = write_dataset(args.out, args.data_root, pid, idx, res)
    if args.qc:
        on_step("qc")
        qc_dir = os.path.join(args.out, "qc")
        os.makedirs(qc_dir, exist_ok=True)
        qc_path = os.path.join(qc_dir, f"{pid}_{idx}_qc.png")
        qc_png(
            itk.array_from_image(res.fu_ref),
            itk.array_from_image(res.warped_img),
            res.warped_seg,
            [c for _, c in res.bl_clicks],
            qc_path,
        )
        paths.append(qc_path)
    return len(res.bl_clicks), int(res.warped_seg.sum()), paths


def _results_table(rows: list[tuple[str, str, str]]) -> Table:
    t = Table(title="registration results", box=None, padding=(0, 1))
    t.add_column("case", style="cyan")
    t.add_column("status")
    t.add_column("detail", style="dim")
    for case_id, status, detail in rows:
        colour = "green" if status == "ok" else "red"
        t.add_row(case_id, f"[{colour}]{status}[/{colour}]", detail)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description="Warp BL→FU frame (itk-elastix) for one or many longitudinal pairs")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pid", help="single case: patient id (with --idx)")
    ap.add_argument("--idx", help="single case: timepoint index (with --pid)")
    ap.add_argument("--sample", type=int, help="process N random BL/FU pairs")
    ap.add_argument("--all", action="store_true", help="process every BL/FU pair")
    ap.add_argument("--seed", type=int, default=0, help="rng seed for --sample")
    ap.add_argument("--qc", action="store_true", help="write axial QC montage PNG")
    ap.add_argument("--no-body-mask", action="store_true", help="disable body-masked metric")
    ap.add_argument("--no-refine", action="store_true", help="disable per-lesion VOI refinement")
    ap.add_argument("--threads", default="auto", help="ITK threads: auto, all, integer, or NANOUNET_REG_THREADS env")
    ap.add_argument("--verbose", action="store_true", help="show elastix/transformix console log")
    args = ap.parse_args()
    assert (args.pid and args.idx) or args.sample or args.all, "pass --pid/--idx, --sample N, or --all"

    nano_header("nanoUNet register-longi", color="magenta")

    threads = apply_threads(resolve_threads(args.threads))
    cases = _select_cases(args)
    os.makedirs(args.out, exist_ok=True)
    rows: list[tuple[str, str, str]] = []

    with _quiet_stderr(args.verbose), Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=32),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=_CONSOLE,
    ) as prog:
        task = prog.add_task("cases", total=len(cases))
        for pid, idx in cases:
            case_id = f"{pid}_{idx}"

            def on_step(name: str) -> None:
                prog.update(task, description=f"{case_id}  {name}")

            t0 = time.monotonic()
            try:
                n_clicks, warped_vox, _ = _process(args, pid, idx, threads, on_step)
            except (AssertionError, RuntimeError) as e:
                rows.append((case_id, "skip", str(e).splitlines()[-1][:120]))
            else:
                dt = time.monotonic() - t0
                rows.append((case_id, "ok", f"{n_clicks} clicks, {warped_vox:,} vox, {dt:.1f}s"))
            prog.advance(task)

    cprint(_results_table(rows))
    n_skip = sum(1 for _, s, _ in rows if s != "ok")
    if n_skip:
        cprint(f"[red]{n_skip}/{len(rows)} case(s) skipped[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
