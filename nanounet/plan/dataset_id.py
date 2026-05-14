"""Resolve dataset id → folder name (DatasetXXX_*); build raw train image/label paths like nnU-Net."""

from __future__ import annotations

import os
import re
from batchgenerators.utilities.file_and_folder_operations import isdir, join, load_json, subdirs, subfiles

from nanounet.common import preprocessed_dir, raw_dir, results_dir, sync_nnunet_env


def convert_id_to_dataset_name(dataset_id: int) -> str:
    sync_nnunet_env()
    prefix = f"Dataset{dataset_id:03d}"
    found: list[str] = []
    for base in (preprocessed_dir(), raw_dir(), results_dir()):
        if isdir(base):
            found.extend(subdirs(base, prefix=prefix, join=False))
    u = sorted(set(found))
    if len(u) > 1:
        raise RuntimeError(f"ambiguous dataset id {dataset_id}: {u}")
    if not u:
        raise RuntimeError(f"no folder starting with {prefix!r} under raw/preprocessed/results")
    return u[0]


def _identifiers_images_tr(raw_folder: str, file_ending: str) -> list[str]:
    files = subfiles(join(raw_folder, "imagesTr"), suffix=file_ending, join=False)
    crop = len(file_ending) + 5
    return sorted(set(i[:-crop] for i in files))


def _paths_for_id(folder: str, files: list[str], file_ending: str, ident: str) -> list[str]:
    p = re.compile(re.escape(ident) + r"_\d\d\d\d" + re.escape(file_ending))
    return [join(folder, i) for i in files if p.fullmatch(i)]


def get_filenames_of_train_images_and_targets(raw_dataset_folder: str, dataset_json: dict | None = None) -> dict:
    if dataset_json is None:
        dataset_json = load_json(join(raw_dataset_folder, "dataset.json"))
    if "dataset" in dataset_json:
        d = dataset_json["dataset"]
        for k in d:
            lf = os.path.expandvars(d[k]["label"])
            d[k]["label"] = os.path.abspath(join(raw_dataset_folder, lf)) if not os.path.isabs(lf) else lf
            d[k]["images"] = [
                os.path.abspath(join(raw_dataset_folder, os.path.expandvars(i))) if not os.path.isabs(os.path.expandvars(i)) else os.path.expandvars(i)
                for i in d[k]["images"]
            ]
        return d
    fe = dataset_json["file_ending"]
    ids = _identifiers_images_tr(raw_dataset_folder, fe)
    img_dir = join(raw_dataset_folder, "imagesTr")
    all_files = subfiles(img_dir, suffix=fe, join=False, sort=True)
    images = [_paths_for_id(img_dir, all_files, fe, i) for i in ids]
    segs = [join(raw_dataset_folder, "labelsTr", i + fe) for i in ids]
    return {i: {"images": im, "label": se} for i, im, se in zip(ids, images, segs)}
