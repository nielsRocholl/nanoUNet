"""Materialize cached dense v8_native PyG graphs for one tracking split."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset

from core.ui import arg_rows
from lesionglue.common import CACHE_ROOT, DATASET_ROOT, config_table, cprint, nano_header
from lesionglue.data.cache.dataset import LesionDataset
from lesionglue.data.features.layout import CACHE_TAG
from lesionglue.data.graph.dense import GraphConfig
from lesionglue.data.source.propagate import PROP_FILLS

SPLITS = ("train", "val", "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATASET_ROOT), help="Longitudinal-CT dataset root holding the raw cases")
    ap.add_argument("--cache", default=str(CACHE_ROOT), help="output root for the cached graphs (written under processed/)")
    ap.add_argument("--split", choices=[*SPLITS, "all"], required=True, help="which tracking split to build, or all for train, val and test")
    ap.add_argument("--k-intra", type=int, default=8, help="neighbors per node in the intra-timepoint kNN graph")
    ap.add_argument("--jobs", type=int, default=1, help="parallel patients (ProcessPool)")
    ap.add_argument("--keep-unclear", action="store_true", help="keep lesions whose link the annotators flagged linking_unclear (default drops them); needs its own --cache dir, e.g. /nnunet_data/lesion_tracking/cache_v9_unclear")
    ap.add_argument("--prop-fill", choices=PROP_FILLS, default="none", help="BL lesions without cog_propagated (129) are dropped (none); unigradicon gives them the registration's bl_click where its sanity_ok is true (needs its own --cache dir)")
    ap.add_argument("--resume", action="store_true", help="keep already-built patients in the staging dir and build only the missing ones; default rebuilds the split")
    args = ap.parse_args()
    nano_header("lesionglue_preprocess")
    config_table(arg_rows(ap, args))

    cache = Path(args.cache)
    problems = []
    settings = {"keep_unclear": args.keep_unclear, "prop_fill": args.prop_fill}  # recorded in each split's *_meta.pt when not default
    if settings != {"keep_unclear": False, "prop_fill": "none"} and cache == CACHE_ROOT:
        problems.append(
            f"--keep-unclear / --prop-fill would write into the default cache {cache}.\n"
            "Expected a separate cache dir, so graphs built with different settings never mix.\n"
            "Fix: --cache /nnunet_data/lesion_tracking/cache_v9"
        )
    for meta in sorted((cache / "processed").glob(f"*_{CACHE_TAG}_meta.pt")):
        built = torch.load(meta)
        for key, want in settings.items():
            got = built.get(key, False if key == "keep_unclear" else "none")
            if got != want:
                problems.append(
                    f"{meta.name} in {cache} was built with {key}={got}, this run has {key}={want}.\n"
                    "Expected every split of one cache dir to be built with the same settings.\n"
                    f"Fix: match --{key.replace('_', '-')} to that cache, or use a new --cache dir"
                )
    if problems:
        raise SystemExit(f"{len(problems)} problem(s) with --keep-unclear, --prop-fill and --cache\nExpected one setting per cache dir, and a dir of its own for non-default settings.\nFix: apply the Fix line of each problem below\n" + "\n".join(problems))
    cache.mkdir(parents=True, exist_ok=True)
    stale = list((cache / "processed").glob("*_v5_l0.pt")) if (cache / "processed").is_dir() else []
    if stale:
        cprint(f"[yellow]stale v5_l0 cache ignored (using {CACHE_TAG}): {stale[0].parent}[/yellow]")
        cprint("[dim]delete those files if they confuse you; they are not loaded[/dim]")

    root = Path(args.root)
    cfg = GraphConfig(k_intra=args.k_intra)
    splits = SPLITS if args.split == "all" else (args.split,)
    counts = {}
    for split in splits:
        ds = LesionDataset(
            root=str(cache), split=split, dataset_root=root, cfg=cfg,
            num_workers=args.jobs, resume=args.resume, keep_unclear=args.keep_unclear, prop_fill=args.prop_fill,
        )
        counts[split] = len(ds)
        graphs = [InMemoryDataset.get(ds, i) for i in range(len(ds))]
        nodes = {nt: sum(g[nt].num_nodes for g in graphs) for nt in ("bl", "fu")}
        cprint(f"built {counts[split]} graphs split={split} -> {cache}/processed/{split}_{CACHE_TAG}.pt")
        cprint(
            f"  nodes bl={nodes['bl']} fu={nodes['fu']} | BL nodes filled from uniGradICON: {sum(int(g['bl'].prop_source.sum()) for g in graphs)}"
            f" | BL lesions left without cog_propagated (no node): {sum(int(g.n_bl_no_prop) for g in graphs)}"
        )
    if len(splits) > 1:
        parts = ", ".join(f"{s}={counts[s]}" for s in splits)
        cprint(f"done -> {cache}: {parts}")
    cprint(f"next: lesionglue_train --config lesionglue/configs/base.json --cache {shlex.quote(str(cache))} --out runs/base", markup=False, soft_wrap=True)


if __name__ == "__main__":
    main()
