"""Multiprocess dataset fingerprint → ``dataset_fingerprint.json`` (intensity stats, spacings, shapes)."""

from __future__ import annotations

import multiprocessing
from time import sleep
from typing import Type

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json, maybe_mkdir_p, save_json

from nanounet.common import cprint, nano_progress, preprocessed_dir, raw_dir
from nanounet.data.crop import crop_to_nonzero
from nanounet.data.io import reader_writer_class_from_dataset
from nanounet.plan.dataset_id import convert_id_to_dataset_name, get_filenames_of_train_images_and_targets


def _collect_fg_intensity(seg: np.ndarray, images: np.ndarray, seed: int, num_samples: int):
    assert images.ndim == 4 and seg.ndim == 4
    rs = np.random.RandomState(seed)
    fg = seg[0] > 0
    per_ch, stats = [], []
    pct = np.array((0.5, 50.0, 99.5))
    for i in range(len(images)):
        px = images[i][fg]
        n = len(px)
        per_ch.append(rs.choice(px, num_samples, replace=True) if n > 0 else np.array([], dtype=np.float32))
        if n > 0:
            p0, med, p99 = np.percentile(px, pct)
            stats.append(
                {
                    "mean": float(np.mean(px)),
                    "median": float(med),
                    "min": float(np.min(px)),
                    "max": float(np.max(px)),
                    "percentile_99_5": float(p99),
                    "percentile_00_5": float(p0),
                }
            )
        else:
            stats.append({k: float("nan") for k in ("mean", "median", "min", "max", "percentile_99_5", "percentile_00_5")})
    return per_ch, stats


def _analyze_case(image_files: list[str], seg_file: str, rw_cls: Type, num_samples: int):
    rw = rw_cls()
    images, prop_im = rw.read_images(image_files)
    seg, _ = rw.read_seg(seg_file)
    dc, sc, bbox = crop_to_nonzero(images, seg)
    per_ch, _ = _collect_fg_intensity(sc, dc, 1234, num_samples)
    sp = prop_im["spacing"]
    sb = images.shape[1:]
    sa = dc.shape[1:]
    rel = np.prod(sa) / np.prod(sb)
    return sa, sp, per_ch, rel


def run_fingerprint(dataset_id: int, num_proc: int, clean: bool = True, foreground_voxels_budget: float = 10e7) -> dict:
    dn = convert_id_to_dataset_name(dataset_id)
    inp = join(raw_dir(), dn)
    outp = join(preprocessed_dir(), dn)
    dj = load_json(join(inp, "dataset.json"))
    ds = get_filenames_of_train_images_and_targets(inp, dj)
    maybe_mkdir_p(outp)
    fp_path = join(outp, "dataset_fingerprint.json")
    if isfile(fp_path) and not clean:
        return load_json(fp_path)
    k0 = next(iter(ds.keys()))
    rw_cls = reader_writer_class_from_dataset(dj, ds[k0]["images"][0], verbose=True)
    n_samp = max(1, int(foreground_voxels_budget // len(ds)))
    r = []
    with nano_progress(len(ds), f"fingerprint {len(ds)} cases") as advance:
        with multiprocessing.get_context("spawn").Pool(num_proc) as pool:
            for key in ds.keys():
                r.append(pool.starmap_async(_analyze_case, ((ds[key]["images"], ds[key]["label"], rw_cls, n_samp),)))
            rem = list(range(len(ds)))
            w = [j for j in pool._pool]
            while rem:
                if not all(j.is_alive() for j in w):
                    raise RuntimeError("worker died (OOM?)")
                done = [i for i in rem if r[i].ready()]
                for _ in done:
                    r[_].get()
                rem = [i for i in rem if i not in done]
                if done:
                    advance(len(done))
                sleep(0.05)
    results = [x.get()[0] for x in r]
    shapes = [x[0] for x in results]
    spacings = [x[1] for x in results]
    fi = [np.concatenate([x[2][i] for x in results]) for i in range(len(results[0][2]))]
    fi = np.array(fi)
    modalities = dj["channel_names"] if "channel_names" in dj else dj["modality"]
    nch = len(modalities.keys())
    pct = np.array((0.5, 50.0, 99.5))
    intensity_ch = {}
    for i in range(nch):
        p0, med, p99 = np.percentile(fi[i], pct)
        intensity_ch[i] = {
            "mean": float(np.mean(fi[i])),
            "median": float(med),
            "std": float(np.std(fi[i])),
            "min": float(np.min(fi[i])),
            "max": float(np.max(fi[i])),
            "percentile_99_5": float(p99),
            "percentile_00_5": float(p0),
        }
    med_rel = float(np.median([x[3] for x in results]))
    fingerprint = {
        "spacings": spacings,
        "shapes_after_crop": shapes,
        "foreground_intensity_properties_per_channel": intensity_ch,
        "median_relative_size_after_cropping": med_rel,
    }
    save_json(fingerprint, fp_path)
    cprint("[bold green]✓ wrote dataset_fingerprint.json[/bold green]")
    return fingerprint
