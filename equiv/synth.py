"""Stage A: deterministic synthetic raw tree for the equivalence harness (temporary, see plan §8).

Two raw datasets (901, 902), 4 cases each, anisotropic spacing that varies per case (hits the
resampling + aniso DA paths), 2-4 ellipsoid lesions per case, fixed-seed noise. Also writes:
  - an instance-labeled GT folder + points JSONs for nanounet_predict --gt-dir (score.py)
  - lesion-type CSVs for nanounet_lesion_weights (d901 ids carry the _BL_img/_FU_img markers)
  - a synthetic registration error table + copies of configs/*.json pointing at it (the real
    configs point at /nnunet_data/..., which does not exist off-cluster)
Everything is a pure function of the seed, so two runs write byte-identical inputs."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import SimpleITK as sitk

SHAPE = (56, 64, 96)  # (Z, Y, X): lesions live in Z < 14 and X < 30, so many patches are lesion-free (decoys)
SPACINGS_ZYX = [(2.5, 0.8, 0.8), (3.0, 0.8, 0.8), (2.5, 0.75, 0.75), (2.75, 0.85, 0.8)]
CASES = {
    901: ["LCT_a0_BL_img", "LCT_a0_FU_img", "LCT_a1_BL_img", "LCT_a1_FU_img"],
    902: ["case_000", "case_001", "case_002", "case_003"],
}
NAMES = {901: "Dataset901_SynthA", 902: "Dataset902_SynthB"}
TYPES = ["Lymph node", "Lung", "Skeleton", "Soft tissue / Skin"]


def _lesions(rng: np.random.Generator) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    out = []
    n = int(rng.integers(2, 5))
    while len(out) < n:
        r = (int(rng.integers(2, 4)), int(rng.integers(3, 6)), int(rng.integers(3, 6)))
        c = (int(rng.integers(r[0] + 1, 14 - r[0])), int(rng.integers(r[1] + 1, SHAPE[1] - r[1] - 1)),
             int(rng.integers(r[2] + 1, 30 - r[2])))
        if all(sum((a - b) ** 2 for a, b in zip(c, o)) ** 0.5 > 9 for o, _ in out):
            out.append((c, r))
    return out


def _case(rng: np.random.Generator):
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in SHAPE], indexing="ij")
    img = rng.normal(-100.0, 40.0, SHAPE).astype(np.float32)
    inst = np.zeros(SHAPE, dtype=np.uint8)
    les = _lesions(rng)
    for i, (c, r) in enumerate(les, 1):
        m = ((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2 <= 1.0
        inst[m] = i
        img[m] += 150.0
    body = ((yy - SHAPE[1] / 2) / (SHAPE[1] / 2.1)) ** 2 + ((xx - SHAPE[2] / 2) / (SHAPE[2] / 2.1)) ** 2 <= 1.0
    img[~body] = -1000.0
    return img, inst, les


def _write(arr: np.ndarray, spacing_zyx, path: str) -> None:
    im = sitk.GetImageFromArray(arr)
    im.SetSpacing(tuple(float(s) for s in spacing_zyx[::-1]))
    im.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(im, path, useCompression=True)


def _error_table(path: str) -> None:
    rng = np.random.default_rng(7)
    bins = [[0.0, 10.0], [10.0, 20.0], [20.0, 1000.0]]
    backends = {b: {"offsets_zyx": [np.round(rng.normal(0, 2, (6, 3)), 2).tolist() for _ in bins]}
                for b in ("original", "unigradicon")}
    doc = {"frame": "resampled", "spacing_zyx": [2.5, 0.8, 0.8], "size_bins_mm": bins,
           "backends": backends, "excluded": [], "provenance": "equiv synth"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


def main(root: str, repo: str) -> None:
    raw = os.path.join(root, "raw")
    extra = os.path.join(root, "extra")
    for d in ("raw", "preprocessed", "results", "tmp", "extra/gt", "extra/predin", "extra/meta", "extra/configs"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    rng = np.random.default_rng(1234)
    for did, ids in CASES.items():
        base = os.path.join(raw, NAMES[did])
        os.makedirs(os.path.join(base, "imagesTr"), exist_ok=True)
        os.makedirs(os.path.join(base, "labelsTr"), exist_ok=True)
        rows: dict[str, list] = {}
        for k, cid in enumerate(ids):
            sp = SPACINGS_ZYX[k]
            img, inst, les = _case(rng)
            _write(img, sp, os.path.join(base, "imagesTr", cid + "_0000.nii.gz"))
            _write((inst > 0).astype(np.uint8), sp, os.path.join(base, "labelsTr", cid + ".nii.gz"))
            if did == 901:
                h, tp = cid.split("_")[1], cid.split("_")[2]
                for i, (c, _) in enumerate(les):
                    xyz = f"{c[2]} {c[1]} {c[0]}"
                    rows.setdefault(h, []).append({"lesion_type": TYPES[i % len(TYPES)],
                                                   "cog_bl": xyz if tp == "BL" else "",
                                                   "cog_fu": xyz if tp == "FU" else ""})
            if did == 902 and k == 0:  # predict/score input: native scan + sibling clicks + instance GT
                _write(img, sp, os.path.join(extra, "predin", "caseA.nii.gz"))
                _write(inst, sp, os.path.join(extra, "gt", "caseA.nii.gz"))
                pts = [{"name": str(i), "point": [float(c[2]), float(c[1]), float(c[0])]} for i, (c, _) in enumerate(les, 1)]
                with open(os.path.join(extra, "predin", "caseA.json"), "w", encoding="utf-8") as f:
                    json.dump({"points": pts, "type": "Multiple points"}, f)
        dj = {"channel_names": {"0": "CT"}, "labels": {"background": 0, "lesion": 1},
              "numTraining": len(ids), "file_ending": ".nii.gz"}
        if did == 901:
            dj["lesion_site"] = "thorax"
        with open(os.path.join(base, "dataset.json"), "w", encoding="utf-8") as f:
            json.dump(dj, f, indent=2)
        for h, rr in rows.items():
            with open(os.path.join(extra, "meta", h + ".csv"), "w", encoding="utf-8") as f:
                f.write("lesion_type,cog_bl,cog_fu\n")
                for r in rr:
                    f.write(f"{r['lesion_type']},{r['cog_bl']},{r['cog_fu']}\n")
    table = os.path.join(extra, "error_table.json")
    _error_table(table)
    for name in ("default.json", "instance_conditional.json"):
        with open(os.path.join(repo, "configs", name), encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["sampling"]["propagated"]["error_table"] = table
        with open(os.path.join(extra, "configs", name), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    gauss = json.loads(open(os.path.join(extra, "configs", "default.json"), encoding="utf-8").read())
    gauss["sampling"]["propagated"]["mode"] = "gaussian"
    with open(os.path.join(extra, "configs", "gaussian.json"), "w", encoding="utf-8") as f:
        json.dump(gauss, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
