"""Spawn-pool case preprocess, GT copy into preprocessed tree, optional centroid JSON."""

from __future__ import annotations

import multiprocessing
import os
import shutil
from time import sleep
from typing import Optional

from batchgenerators.utilities.file_and_folder_operations import isdir, isfile, join, load_json, maybe_mkdir_p

from nanounet.common import cprint, nano_progress, preprocessed_dir, raw_dir
from nanounet.plan.prep.case_pp import run_case_save
from nanounet.plan.dataset_id import convert_id_to_dataset_name, get_filenames_of_train_images_and_targets
from nanounet.plan.plans import Plans
from nanounet.prompt.centroids import precompute_folder


def _worker(args: tuple):
    out_t, imgs, lab, plans_f, cfg, dj_f, verb = args
    pl = Plans(plans_f)
    run_case_save(out_t, imgs, lab, pl, pl.get_configuration(cfg), load_json(dj_f), verbose=verb)


def _cgroup_mem_limit_gb() -> Optional[float]:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(p, encoding="utf-8") as f:
                raw = f.read().strip()
        except OSError:
            continue
        if raw == "max":
            continue
        try:
            v = int(raw)
        except ValueError:
            continue
        if v < (1 << 40) * 1024:  # exclude the "no limit" sentinel some kernels report as a huge int
            return v / 1e9
    return None


def _cgroup_oom_kills() -> Optional[int]:
    try:
        with open("/sys/fs/cgroup/memory.events", encoding="utf-8") as f:
            for line in f:
                if line.startswith("oom_kill "):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def _dead_worker_error(num_processes: int, resume_flag: str) -> RuntimeError:
    lim_gb = _cgroup_mem_limit_gb()
    oom_kills = _cgroup_oom_kills()
    lines = [
        "A preprocess worker was killed with no Python exception (SIGKILL) -- almost always an "
        "out-of-memory (OOM) kill by the cgroup, not a bug in the preprocessing code.",
    ]
    if lim_gb is not None:
        per_worker = lim_gb / num_processes
        lines.append(
            f"cgroup memory limit: {lim_gb:.1f} GB across {num_processes} workers = "
            f"{per_worker:.1f} GB/worker."
        )
    else:
        lines.append(f"{num_processes} workers configured (cgroup memory limit unreadable).")
    if oom_kills:
        lines.append(f"cgroup memory.events reports {oom_kills} oom_kill event(s) so far.")
    lines.append(
        "Large volumes can peak at ~50 GB/worker during resampling, so a worker count sized for "
        "average cases can still get OOM-killed on the largest ones."
    )
    lines.append(f"Fix: rerun with fewer workers: nanounet_preprocess ... -np {max(1, num_processes // 2)}{resume_flag}")
    return RuntimeError("\n".join(lines))


def run_preprocess(
    dataset_id: int,
    plans_identifier: str,
    num_processes: int,
    resume: bool,
    configuration: str = "3d_fullres",
    verbose: bool = True,
    sidecars_only: bool = False,
) -> None:
    dn = convert_id_to_dataset_name(dataset_id)
    raw = join(raw_dir(), dn)
    pre = join(preprocessed_dir(), dn)
    assert isdir(raw)
    plans_f = join(pre, plans_identifier + ".json")
    assert isfile(plans_f), plans_f
    dj_raw = load_json(join(raw, "dataset.json"))
    assert isfile(join(pre, "dataset.json"))
    pl = Plans(plans_f)
    cm = pl.get_configuration(configuration)
    out_dir = join(pre, cm.data_identifier)
    if sidecars_only:
        if not os.path.isdir(out_dir):
            raise FileNotFoundError(
                f"--sidecars-only needs an existing preprocessed folder at {out_dir}, but it is missing.\n"
                f"Expected output of a prior preprocess run for dataset {dataset_id}, plans {plans_identifier}.\n"
                f"Fix: run without --sidecars-only first: "
                f"nanounet_preprocess -d {dataset_id} --planner nnUNetPlannerResEncL -np {num_processes}"
            )
        precompute_folder(out_dir, num_processes, resume=False)
        if verbose:
            cprint(f"[bold green]✓ sidecars regenerated → {out_dir}[/bold green]")
        return
    if not resume and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    maybe_mkdir_p(out_dir)
    ds = get_filenames_of_train_images_and_targets(raw, dj_raw)

    def _done(k: str) -> bool:
        b = join(out_dir, k)
        return isfile(b + ".b2nd") and isfile(b + "_seg.b2nd") and isfile(b + ".pkl")

    todo = {k: v for k, v in ds.items() if not _done(k)}
    dj_path = join(pre, "dataset.json")
    if not todo:
        cprint("[dim]all cases complete[/dim]")
    else:
        r = []
        with nano_progress(len(todo), f"preprocess {len(todo)} cases") as advance:
            with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
                for k in todo:
                    r.append(
                        pool.apply_async(
                            _worker,
                            (
                                (
                                    join(out_dir, k),
                                    todo[k]["images"],
                                    todo[k]["label"],
                                    plans_f,
                                    configuration,
                                    dj_path,
                                    False,
                                ),
                            ),
                        )
                    )
                rem = list(range(len(r)))
                w = [j for j in pool._pool]
                while rem:
                    if not all(j.is_alive() for j in w):
                        raise _dead_worker_error(num_processes, " --resume")
                    done_ix = [i for i in rem if r[i].ready()]
                    for i in done_ix:
                        r[i].get()
                    if done_ix:
                        advance(len(done_ix))
                    rem = [i for i in rem if i not in done_ix]
                    sleep(0.05)
    maybe_mkdir_p(join(pre, "gt_segmentations"))
    fe = dj_raw["file_ending"]
    for k in ds:
        shutil.copyfile(ds[k]["label"], join(pre, "gt_segmentations", k + fe))
    precompute_folder(out_dir, num_processes, resume)
    if verbose:
        cprint(f"[bold green]✓ {len(ds)} cases → {out_dir}[/bold green]")
