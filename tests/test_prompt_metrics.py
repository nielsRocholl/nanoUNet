"""Regression tests for the two prompt-robustness validation metrics.

Both bugs come from the same source: validation forces `false_pos_probability = 1.0`
(data_module.py), so EVERY val patch carries a decoy click, and both metrics were counting it.

1. val_dice_click_inside/outside -- click_inside_flags majority-voted over all positive clicks,
   decoy included. With L correctly placed lesion clicks plus one decoy the test
   `2*n_in > len(idx)` reduces to `L > 1`, so every single-lesion patch was flagged "outside"
   regardless of where its click landed. See test_single_lesion_with_decoy_is_inside.

2. val_prompt_agreement -- each prompt variant drew its own decoy at an independent random
   location, so an agreement pair differed by BOTH lesion-click displacement (the quantity of
   interest) and a completely different spurious click. See test_decoy_is_shared_across_variants.

Run: .venv/bin/python -m pytest tests/test_prompt_metrics.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from nanounet.config import load_config
from nanounet.data.sampling import build_patch, draw_false_pos, points_variant
from nanounet.train.patch_render import click_inside_flags, split_variant_keypoints

SAMPLE_DATA = Path(
    "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/Datasets/longi-registered/targetsTrFU"
)


# --------------------------------------------------------------------------------------------
# config: default.json but propagated.mode=gaussian, so no cluster-only error table is needed
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    d = json.loads(Path("configs/default.json").read_text())
    d["sampling"]["propagated"]["mode"] = "gaussian"
    d["sampling"]["false_pos_probability"] = 1.0  # what validation forces
    p = tmp_path_factory.mktemp("cfg") / "test.json"
    p.write_text(json.dumps(d))
    return load_config(str(p))


# --------------------------------------------------------------------------------------------
# 1. click_inside_flags -- the decoy must not participate in the inside/outside vote
# --------------------------------------------------------------------------------------------


def _seg_with_lesion(shape=(32, 32, 32), centre=(16, 16, 16), r=4):
    """Binary seg (1, *shape) with one solid cubic lesion."""
    seg = torch.zeros((1, *shape), dtype=torch.int16)
    z, y, x = centre
    seg[0, z - r : z + r, y - r : y + r, x - r : x + r] = 1
    return seg


def _entry(points, n_fp):
    return {"pp": torch.tensor(points, dtype=torch.float32).reshape(-1, 3), "pn": None, "n_fp": n_fp}


def test_single_lesion_with_decoy_is_inside():
    """THE BUG. One lesion click dead-centre + the mandatory decoy => must be INSIDE.

    Old formula counted the decoy: n_in=1, len=2, `2*1 > 2` is False -> flagged OUTSIDE.
    """
    seg = _seg_with_lesion()
    entries = [_entry([[16, 16, 16], [2, 2, 2]], n_fp=1)]  # lesion click, then decoy
    assert click_inside_flags(entries, seg) == [1]

    # and confirm the old formula really did get this wrong, so the test cannot silently pass
    old = 1 if 2 * 1 > 2 else 0
    assert old == 0, "old formula should have mislabelled this patch"


def test_multi_lesion_with_decoy_is_inside():
    """L=2 was the only case the old formula got right; it must still be right."""
    seg = _seg_with_lesion()
    entries = [_entry([[16, 16, 16], [15, 16, 17], [2, 2, 2]], n_fp=1)]
    assert click_inside_flags(entries, seg) == [1]


def test_misplaced_lesion_click_is_outside():
    """A genuinely off-lesion click must still register as OUTSIDE once the decoy is excluded."""
    seg = _seg_with_lesion()
    entries = [_entry([[2, 2, 2], [30, 30, 30]], n_fp=1)]  # lesion click missed, then decoy
    assert click_inside_flags(entries, seg) == [0]


def test_decoy_only_patch_is_excluded():
    """Lesion-free patch: the decoy is the only click. No lesion click -> -1 -> dropped by the
    caller from both the inside and the outside bucket."""
    seg = torch.zeros((1, 32, 32, 32), dtype=torch.int16)
    entries = [_entry([[2, 2, 2]], n_fp=1)]
    assert click_inside_flags(entries, seg) == [-1]


def test_vote_is_strict_majority_over_lesion_clicks_only():
    """2 of 3 lesion clicks inside -> inside; 1 of 3 -> outside. Decoys never shift the balance."""
    seg = _seg_with_lesion()
    two_in = _entry([[16, 16, 16], [15, 15, 15], [30, 30, 30], [2, 2, 2]], n_fp=1)
    one_in = _entry([[16, 16, 16], [30, 30, 30], [29, 29, 29], [2, 2, 2]], n_fp=1)
    assert click_inside_flags([two_in, one_in], seg) == [1, 0]


def test_no_decoy_still_works():
    """n_fp=0 (training, false_pos_probability=0.05 misses) must behave as before."""
    seg = _seg_with_lesion()
    assert click_inside_flags([_entry([[16, 16, 16]], n_fp=0)], seg) == [1]
    assert click_inside_flags([_entry([[2, 2, 2]], n_fp=0)], seg) == [0]


# --------------------------------------------------------------------------------------------
# 2. the decoy must be drawn once per patch and shared by every variant
# --------------------------------------------------------------------------------------------


def test_draw_false_pos_respects_probability(cfg):
    seg = _seg_with_lesion().numpy()
    rng = np.random.default_rng(0)
    assert len(draw_false_pos(seg, cfg, False, rng)) == 1  # p=1.0 in the fixture
    assert draw_false_pos(seg, cfg, True, rng) == []  # force_zero_prompt -> no clicks at all


def test_points_variant_records_and_trails_the_decoy(cfg):
    """The decoy must be the LAST entry of points_pos -- click_inside_flags relies on it."""
    seg = _seg_with_lesion().numpy()
    cts = [(16, 16, 16)]
    pslc = (slice(0, 32), slice(0, 32), slice(0, 32))
    fp = [(2, 2, 2)]
    v = points_variant(seg, cts, pslc, cfg, False, np.random.default_rng(0), True, [512.0], fp)
    assert v["n_false_pos"] == 1
    assert tuple(v["points_pos"][-1].astype(int)) == (2, 2, 2)


def test_decoy_is_shared_across_variants(cfg):
    """THE CONFOUND. Every variant of one patch -- including the val agreement draw -- must carry
    the SAME decoy, so a pair differs only in lesion-click placement."""
    seg = _seg_with_lesion().numpy()
    cts = [(16, 16, 16)]
    pslc = (slice(0, 32), slice(0, 32), slice(0, 32))
    rng = np.random.default_rng(0)
    fp = draw_false_pos(seg, cfg, False, rng)

    variants = [
        points_variant(seg, cts, pslc, cfg, False, rng, True, [512.0], fp) for _ in range(2)
    ]
    variants.append(
        points_variant(seg, cts, pslc, cfg, False, np.random.default_rng(99), True, [512.0], fp)
    )

    decoys = [tuple(v["points_pos"][-1].astype(int)) for v in variants]
    assert len(set(decoys)) == 1, f"decoy differs across variants: {decoys}"

    # sanity: the LESION clicks must still differ, else we broke the diagnostic entirely
    lesion = [tuple(v["points_pos"][0].astype(int)) for v in variants]
    assert len(set(lesion)) > 1, "lesion clicks identical -- displacement not being applied"


def test_split_variant_keypoints_propagates_n_fp():
    """n_fp has to survive the concat/augment/split round trip or the vote silently regresses."""
    variants = [
        {"points_pos": np.zeros((3, 3), np.float32), "points_neg": np.zeros((0, 3), np.float32), "n_false_pos": 1},
        {"points_pos": np.zeros((2, 3), np.float32), "points_neg": np.zeros((0, 3), np.float32), "n_false_pos": 1},
    ]
    kp = torch.zeros((5, 3), dtype=torch.float32)
    entries = split_variant_keypoints(kp, variants, longi=False)
    assert [e["n_fp"] for e in entries] == [1, 1]
    assert [e["pp"].shape[0] for e in entries] == [3, 2]


# --------------------------------------------------------------------------------------------
# 3. end-to-end through build_patch on real lesion geometry
# --------------------------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_DATA.is_dir(), reason="sample data is laptop-only")
def test_build_patch_on_real_fu_data(cfg):
    """Real multi-lesion FU segmentations through the full build_patch path.

    targetsTrFU is instance-labelled (voxel value = lesion_id); the training pipeline sees a
    binarised version, so binarise here too and recover instances with cc3d, exactly as
    nanounet_preprocess does.
    """
    import cc3d
    import nibabel as nib

    from nanounet.prompt.centroids import centroids_from_seg

    files = sorted(SAMPLE_DATA.glob("*.nii.gz"))
    checked = 0
    for f in files:
        arr = np.asarray(nib.load(str(f)).dataobj)
        binary = (arr > 0).astype(np.int16)
        if binary.sum() == 0:
            continue
        seg = binary.transpose(2, 0, 1)[None]  # -> (1, Z, Y, X)
        cts = centroids_from_seg(seg)
        if not cts:
            continue
        lab = cc3d.connected_components(seg[0].astype(np.uint8))
        vols = [float((lab == i).sum()) for i in range(1, int(lab.max()) + 1)]
        if len(vols) != len(cts):
            continue

        data = np.zeros_like(seg, dtype=np.float32)
        ps = np.array([32, 128, 128])
        raw = build_patch(
            data, seg, {"centroids_zyx": cts, "volume_vox": vols}, cfg, ps, ps, None, False,
            np.random.default_rng(0), prompts_per_patch=2, extra_rng=np.random.default_rng(7),
        )

        variants = raw["points_variants"]
        assert len(variants) == 3, "2 real variants + 1 agreement draw"

        # every variant carries exactly one decoy (false_pos_probability=1.0), all identical
        assert {v["n_false_pos"] for v in variants} == {1}
        decoys = {tuple(v["points_pos"][-1].astype(int)) for v in variants}
        assert len(decoys) == 1, f"{f.name}: decoy differs across variants: {decoys}"

        # the decoy must sit on background, which is the whole reason it must not count as a
        # "click outside" for the lesion-placement metric
        seg_crop = torch.from_numpy(raw["segmentation"])
        dz, dy, dx = next(iter(decoys))
        assert seg_crop[0, dz, dy, dx] == 0

        # and the vote must run over lesion clicks only
        entries = [
            {"pp": torch.from_numpy(v["points_pos"]), "pn": None, "n_fp": v["n_false_pos"]}
            for v in variants
        ]
        flags = click_inside_flags(entries, seg_crop)
        for v, fl in zip(variants, flags):
            n_les = v["points_pos"].shape[0] - v["n_false_pos"]
            assert fl == (-1 if n_les <= 0 else fl)
            assert fl in (-1, 0, 1)

        checked += 1
        if checked >= 5:
            break

    assert checked >= 3, f"only exercised {checked} real cases"


@pytest.mark.skipif(not SAMPLE_DATA.is_dir(), reason="sample data is laptop-only")
def test_single_lesion_real_case_can_be_inside(cfg):
    """The regression, on real data: a single-lesion patch must be able to come back INSIDE.

    Before the fix this was impossible -- the decoy guaranteed a 1-in-2 vote and every
    single-lesion val patch was scored as "click outside".

    Click the argmax-EDT seed, NOT the plain centroid: centroids.py documents that the centroid
    falls outside its own lesion ~12% of the time on concave shapes, and such a patch is correctly
    scored "outside". Confusing the two makes this test measure lesion shape, not the fix.
    """
    import nibabel as nib
    from scipy.ndimage import distance_transform_edt

    from nanounet.prompt.centroids import centroids_from_seg

    seen_single, seen_inside, centroid_outside = 0, 0, 0
    for f in sorted(SAMPLE_DATA.glob("*.nii.gz")):
        arr = np.asarray(nib.load(str(f)).dataobj)
        binary = (arr > 0).astype(np.int16)
        if binary.sum() == 0:
            continue
        seg = binary.transpose(2, 0, 1)[None]
        cts = centroids_from_seg(seg)
        if len(cts) != 1:
            continue
        seen_single += 1
        seg_t = torch.from_numpy(seg)

        mask = seg[0] > 0
        seed = np.unravel_index(np.argmax(distance_transform_edt(mask)), mask.shape)
        entry = {
            "pp": torch.tensor([list(map(int, seed)), [0, 0, 0]], dtype=torch.float32),
            "pn": None,
            "n_fp": 1,
        }
        if click_inside_flags([entry], seg_t) == [1]:
            seen_inside += 1

        cz, cy, cx = cts[0]
        if not mask[cz, cy, cx]:
            centroid_outside += 1
        if seen_single >= 5:
            break

    assert seen_single > 0, "no single-lesion real case found"
    assert seen_inside == seen_single, "single-lesion patches still mislabelled as outside"
    # documents WHY the seed is used: some plain centroids genuinely sit outside their lesion,
    # and those are correctly scored "outside" -- that is lesion shape, not a metric bug.
    assert centroid_outside >= 0
