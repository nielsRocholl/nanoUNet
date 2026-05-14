"""Combine N raw datasets into one synthetic raw folder with ``dNNN_``-prefixed case keys."""

from __future__ import annotations

from batchgenerators.utilities.file_and_folder_operations import isdir, join, load_json, maybe_mkdir_p, save_json, subdirs

from nanounet.common import preprocessed_dir, raw_dir, results_dir, sync_nnunet_env
from nanounet.plan.dataset_id import convert_id_to_dataset_name, get_filenames_of_train_images_and_targets


def _folders_for_id(dataset_id: int) -> list[str]:
    sync_nnunet_env()
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
        raise ValueError("merged_name must not contain path separators")
    merged_name = merged_name.strip()
    if not merged_name:
        raise ValueError("merged_name empty")
    sid_set = set(source_ids)
    if len(sid_set) != len(source_ids):
        raise RuntimeError("duplicate source dataset ids")
    if merged_id in sid_set:
        raise RuntimeError("--merged-id must not appear in source -d list")
    sync_nnunet_env()
    target = f"Dataset{merged_id:03d}_{merged_name}"
    present = _folders_for_id(merged_id)
    if len(present) > 1:
        raise RuntimeError(f"ambiguous dataset id {merged_id}: {present}")
    if len(present) == 1 and present[0] != target:
        raise RuntimeError(f"--merged-id {merged_id} already used by {present[0]!r}, expected {target!r}")

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
