"""Combine N raw datasets into one synthetic raw folder with ``dNNN_``-prefixed case keys."""

from __future__ import annotations

from batchgenerators.utilities.file_and_folder_operations import isdir, join, load_json, maybe_mkdir_p, save_json, subdirs

from nanounet.common import preprocessed_dir, raw_dir, results_dir
from nanounet.plan.dataset_id import convert_id_to_dataset_name, get_filenames_of_train_images_and_targets


def _folders_for_id(dataset_id: int) -> list[str]:
    prefix = f"Dataset{dataset_id:03d}"
    out: list[str] = []
    for base in (raw_dir(), preprocessed_dir(), results_dir()):
        if isdir(base):
            out.extend(subdirs(base, prefix=prefix, join=False))
    return sorted(set(out))


def _modalities(dj: dict):
    return dj.get("channel_names") or dj["modality"]


def _assert_compatible(ref: dict, dj: dict, src: str) -> None:
    assert ref["file_ending"] == dj["file_ending"], f"file_ending mismatch vs {src}"
    assert ref["labels"] == dj["labels"], f"labels mismatch vs {src}"
    assert _modalities(ref) == _modalities(dj), f"channel_names/modality mismatch vs {src}"


def build_merged_raw(source_ids: list[int], merged_id: int, merged_name: str) -> str:
    """Write merged ``dataset.json`` under raw. Returns folder name ``DatasetXXX_Name``."""
    if any(c in merged_name for c in "/\\"):
        raise ValueError(
            f"--merged-name {merged_name!r} contains a path separator ('/' or '\\\\').\n"
            f"Expected a plain folder-name segment with no '/' or '\\\\'.\n"
            f"Fix: pass --merged-name Merged (or another name with no path separators)   (see docs/steps/preprocess.md)"
        )
    merged_name = merged_name.strip()
    if not merged_name:
        raise ValueError(
            f"--merged-name is empty after stripping whitespace.\n"
            f"Expected a non-empty folder-name segment for the merged dataset (Dataset<id>_<name>).\n"
            f"Fix: pass --merged-name Merged (or another non-empty name)   (see docs/steps/preprocess.md)"
        )
    sid_set = set(source_ids)
    if len(sid_set) != len(source_ids):
        raise RuntimeError(
            f"Duplicate ids in -d {source_ids}.\n"
            f"Expected each -d id to appear exactly once; got {sid_set} unique vs {source_ids} given.\n"
            f"Fix: pass -d with each dataset id once, e.g. -d 1 2 3   (see docs/steps/preprocess.md)"
        )
    if merged_id in sid_set:
        raise RuntimeError(
            f"--merged-id {merged_id} also appears in the source ids {sid_set}.\n"
            f"Expected --merged-id to be a new id, distinct from every source -d id.\n"
            f"Fix: pass a --merged-id not in -d {sid_set}, e.g. --merged-id 999   (see docs/steps/preprocess.md)"
        )
    target = f"Dataset{merged_id:03d}_{merged_name}"
    present = _folders_for_id(merged_id)
    if len(present) > 1:
        raise RuntimeError(
            f"Multiple folders already match --merged-id {merged_id}: {present}.\n"
            f"Expected at most one folder for --merged-id under raw/preprocessed/results.\n"
            f"Fix: remove or rename the extra folder(s), or pick a different --merged-id   (see docs/steps/preprocess.md)"
        )
    if len(present) == 1 and present[0] != target:
        raise RuntimeError(
            f"--merged-id {merged_id} already used by {present[0]!r}, expected {target!r}.\n"
            f"Expected the existing folder for --merged-id {merged_id} to already be named {target!r}.\n"
            f"Fix: pass --merged-name matching {present[0]!r}, or choose a different --merged-id   (see docs/steps/preprocess.md)"
        )

    merged_cases: dict = {}
    sources_meta: list[dict] = []
    ref: dict | None = None
    for sid in source_ids:
        folder_name = convert_id_to_dataset_name(sid)
        raw_path = join(raw_dir(), folder_name)
        dj = load_json(join(raw_path, "dataset.json"))
        if ref is None:
            ref = dj
        else:
            _assert_compatible(ref, dj, folder_name)
        pfx = f"d{sid:03d}_"
        fm = get_filenames_of_train_images_and_targets(raw_path, dj)
        for k, v in fm.items():
            nk = pfx + k
            assert nk not in merged_cases, nk
            merged_cases[nk] = {"images": list(v["images"]), "label": v["label"]}
        sources_meta.append({"id": sid, "name": folder_name, "prefix": pfx, "num_cases": len(fm)})

    assert ref is not None
    out_dir = join(raw_dir(), target)
    maybe_mkdir_p(out_dir)
    out_dj = {k: v for k, v in ref.items() if k != "dataset"}
    key = "channel_names" if "channel_names" in ref else "modality"
    out_dj[key] = _modalities(ref)
    out_dj["numTraining"] = len(merged_cases)
    out_dj["dataset"] = merged_cases
    save_json(out_dj, join(out_dir, "dataset.json"), sort_keys=False)
    save_json(
        {"merged_id": merged_id, "merged_name": merged_name, "sources": sources_meta},
        join(out_dir, "merged_sources.json"),
        sort_keys=False,
    )
    return target
