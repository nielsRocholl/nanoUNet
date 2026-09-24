"""Golden capture (plan §8.2): run stages A-H against one source tree, write {key: digest}.

`python equiv/capture.py --src <tree> --out <dir> --fixtures <dir>`
  --src       checkout whose `nanounet` is imported (a base-SHA worktree, or the working tree)
  --out       writes golden.json + logs/
  --fixtures  old_sup.ckpt / old_mae.ckpt; created from this run's stage F if absent (base run)

CPU only, deterministic: CUDA hidden, PYTHONHASHSEED=0, 1 thread, deterministic algorithms, global
RNGs re-seeded to 0 per stage/sub-step. The synthetic tree lives at a FIXED path
($EQUIV_OUT/tree, wiped per capture) because ckpts and JSONs embed absolute paths (L21).

Excluded by name, with the reason (never silently):"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _digest import digest  # noqa: E402

DS = "Dataset903_Merged"
# macOS caps AF_UNIX socket paths at 104 chars and torch_shm_manager puts its socket in TMPDIR
# (set_safe_tmpdir -> NANOUNET_TMPDIR in DataLoader workers), so the tmp dir must be short.
SHORT_TMP = "/tmp/claude-501/nq"
PLANS = "nnUNetResEncUNetLPlans"

EXCLUDE = {
    # file NAME contains a wall-clock timestamp (cli/build_splits.py backup of the old splits file)
    "splits_final.backup-": "backup filename embeds time.strftime",
    # wall-clock timer, not a computed value
    "epoch_wall_time_sec": "wall-clock seconds (metrics CSV column / logged metric)",
}
# Lightning ckpt keys holding run-environment metadata rather than computed state.
CKPT_SKIP = {
    "pytorch-lightning_version": "library version string, not model state",
}


def _env(src: str, tree: str) -> dict:
    e = dict(os.environ)
    e.update(
        EQUIV_SRC=src, NANOUNET_RAW=f"{tree}/raw", NANOUNET_PREPROCESSED=f"{tree}/preprocessed",
        NANOUNET_RESULTS=f"{tree}/results", NANOUNET_TMPDIR=SHORT_TMP, CUDA_VISIBLE_DEVICES="",
        PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NANOUNET_DL_FORCE_NO_WORKERS="1",
        WANDB_MODE="disabled", COLUMNS="100",
        # macOS: torch_shm_manager is fork()ed from a threaded DataLoader worker; the ObjC runtime
        # aborts such children unless told not to (no effect on Linux or on any computed value).
        OBJC_DISABLE_INITIALIZE_FORK_SAFETY="YES",
    )
    for k in ("WANDB_RUN_ID", "NANOUNET_MEM_DIAG", "PYTHONPATH"):
        e.pop(k, None)
    return e


def _run(name: str, argv: list[str], env: dict, logs: str, cwd: str) -> None:
    t0 = time.time()
    p = subprocess.run([sys.executable, *argv], env=env, cwd=cwd, capture_output=True, text=True, timeout=1200)
    with open(os.path.join(logs, name + ".log"), "w", encoding="utf-8") as f:
        f.write(p.stdout + "\n--- stderr ---\n" + p.stderr)
    print(f"  [{time.time() - t0:5.1f}s] {name}  rc={p.returncode}", file=sys.stderr)
    if p.returncode != 0:
        raise SystemExit(f"stage {name} failed (rc={p.returncode}); see {logs}/{name}.log\n{p.stderr[-3000:]}")


def _cli(name, module, args, env, logs, cwd):
    _run(name, [os.path.join(HERE, "_cli.py"), module, *args], env, logs, cwd)


def _file_digest(path: str):
    import numpy as np

    raw = open(path, "rb").read()
    out = {"bytes": digest(raw)}
    if path.endswith(".json"):
        out["json"] = digest(json.loads(raw))
    elif path.endswith(".pkl"):
        out["pkl"] = digest(pickle.loads(raw))
    elif path.endswith(".npz"):
        z = np.load(path)
        out["npz"] = digest({k: z[k] for k in z.files})
    elif path.endswith(".b2nd"):
        import blosc2

        out["b2nd"] = digest(blosc2.open(path, mode="r")[:])
    elif path.endswith(".nii.gz"):
        import SimpleITK as sitk

        im = sitk.ReadImage(path)
        out["nii"] = digest([sitk.GetArrayFromImage(im), im.GetSpacing(), im.GetOrigin(), im.GetDirection()])
    elif path.endswith(".csv"):
        rows = list(csv.reader(io.StringIO(raw.decode())))
        if rows:
            keep = [i for i, h in enumerate(rows[0]) if h not in EXCLUDE]
            out["csv"] = digest([[r[i] for i in keep if i < len(r)] for r in rows])
            out.pop("bytes")  # raw bytes include the excluded timer column
    elif path.endswith(".ckpt"):
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        for k in CKPT_SKIP:
            ck.pop(k, None)
        out = {"ckpt/" + k: digest(v) for k, v in ck.items()}  # bytes differ: zip + pickle framing
    return out


def _tree_digests(res: dict, root: str, rel_to: str, prefix: str) -> None:
    for p in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if not os.path.isfile(p):
            continue
        rel = os.path.relpath(p, rel_to)
        if any(x in rel for x in EXCLUDE):
            continue
        for k, v in _file_digest(p).items():
            res[f"{prefix}/{rel}::{k}"] = v


def capture(src: str, out: str, fixtures: str, eq_out: str) -> dict:
    tree = os.path.join(eq_out, "tree")
    logs = os.path.join(out, "logs")
    shutil.rmtree(tree, ignore_errors=True)
    os.makedirs(logs, exist_ok=True)
    env = _env(src, tree)
    x = f"{tree}/extra"
    pp = f"{tree}/preprocessed/{DS}"
    res: dict = {}
    t0 = time.time()
    # A synth
    _run("A_synth", [os.path.join(HERE, "synth.py"), tree, src], env, logs, eq_out)
    _tree_digests(res, f"{tree}/raw", tree, "A")
    _tree_digests(res, x, tree, "A")
    # B preprocess (+ merge/fingerprint/plan/spawn pools K10), splits, valset, lesion weights
    _cli("B_preprocess", "nanounet.cli.preprocess",
         ["-d", "901", "902", "--merged-id", "903", "-np", "2", "--gpu-memory-gb", "0.5",
          "--valset-config", f"{x}/configs/default.json", "--valset-n", "80"], env, logs, tree)
    _cli("B_splits", "nanounet.cli.build_splits", ["-d", "903", "--plans", PLANS, "--force"], env, logs, tree)
    _cli("B_valset", "nanounet.cli.build_valset",
         ["-d", "903", "--plans", PLANS, "--config", f"{x}/configs/default.json", "--out", f"{pp}/valset_small.json",
          "--n-patches", "16", "--floor", "4"], env, logs, tree)
    _cli("B_lesion_weights", "nanounet.cli.lesion_weights",
         ["-d", "903", "--plans", PLANS, "--meta-dir", f"{x}/meta", "--only-prefix", "d901_LCT_"], env, logs, tree)
    _tree_digests(res, f"{tree}/preprocessed", tree, "B")
    _tree_digests(res, f"{tree}/raw/{DS}", tree, "B")
    # C/D/E in-process stages
    for st in ("C", "D", "E"):
        j = os.path.join(out, f"stage_{st}.json")
        _run(f"{st}_stage", [os.path.join(HERE, "_stage.py"), st, j], env, logs, tree)
        res.update(json.load(open(j, encoding="utf-8")))
    # F micro-train (sup + EMA + val manifest; integrated MAE; standalone MAE pretrain)
    r = f"{tree}/results"
    common = ["-d", "903", "-f", "0", "--plans", PLANS, "--config", f"{x}/configs/default.json", "--accelerator", "cpu",
              "--precision", "32", "--iters-per-epoch", "2", "--val-iters", "1", "--no-wandb", "--dl-bucket", "s"]
    _cli("F_sup", "nanounet.cli.train", common + ["--epochs", "2", "--ema-decay", "0.999", "--val-manifest",
                                                  f"{pp}/valset_small.json", "--out", f"{r}/F_sup"], env, logs, tree)
    _cli("F_maesup", "nanounet.cli.train", common + ["--epochs", "1", "--mae-pretrain", "--mae-epochs", "1",
                                                     "--mae-iters-per-epoch", "2", "--out", f"{r}/F_maesup"], env, logs, tree)
    _cli("F_pretrain", "nanounet.cli.pretrain",
         ["-d", "903", "-f", "0", "--plans", PLANS, "--epochs", "1", "--iters-per-epoch", "2", "--val-iters", "1",
          "--no-wandb", "--accelerator", "cpu", "--precision", "32", "--dl-bucket", "s", "--out", f"{r}/F_mae"],
         env, logs, tree)
    for d in ("F_sup", "F_maesup", "F_mae"):
        _tree_digests(res, f"{r}/{d}", tree, "F")
    if not os.path.isfile(os.path.join(fixtures, "old_sup.ckpt")):
        os.makedirs(fixtures, exist_ok=True)
        (e0,) = glob.glob(f"{r}/F_sup/checkpoints/best-epoch=0-*.ckpt")
        shutil.copyfile(e0, os.path.join(fixtures, "old_sup.ckpt"))
        shutil.copyfile(f"{r}/F_mae/checkpoints/last.ckpt", os.path.join(fixtures, "old_mae.ckpt"))
        print(f"  fixtures written: {fixtures}", file=sys.stderr)
    # G predict (single: default/--tta/--ema/--disable-tta; folder: --gt-dir scoring)
    g = f"{r}/G"
    for name, extra in (("default", []), ("tta", ["--tta"]), ("ema", ["--ema"]), ("notta", ["--disable-tta"])):
        _cli(f"G_{name}", "nanounet.cli.predict",
             ["-i", f"{x}/predin/caseA.nii.gz", "--points", f"{x}/predin/caseA.json", "-o", f"{g}/{name}/caseA.nii.gz",
              "-m", f"{r}/F_sup", "--device", "cpu", *extra], env, logs, tree)
    _cli("G_score", "nanounet.cli.predict",
         ["-i", f"{x}/predin", "-o", f"{g}/score", "-m", f"{r}/F_sup", "--device", "cpu", "--gt-dir", f"{x}/gt",
          "--metrics-out", f"{g}/score/metrics"], env, logs, tree)
    _tree_digests(res, g, tree, "G")
    # H old-ckpt load (fixtures from the BASE run) + resume
    j = os.path.join(out, "stage_H.json")
    _run("H_stage", [os.path.join(HERE, "_stage.py"), "H", j, fixtures], env, logs, tree)
    res.update(json.load(open(j, encoding="utf-8")))
    os.makedirs(f"{r}/H_resume/checkpoints", exist_ok=True)
    shutil.copyfile(os.path.join(fixtures, "old_sup.ckpt"), f"{r}/H_resume/checkpoints/old_sup.ckpt")
    _cli("H_resume", "nanounet.cli.train", common + ["--epochs", "2", "--ema-decay", "0.999", "--val-manifest",
                                                     f"{pp}/valset_small.json", "--resume",
                                                     f"{r}/H_resume/checkpoints/old_sup.ckpt", "--out", f"{r}/H_resume"],
         env, logs, tree)
    _tree_digests(res, f"{r}/H_resume", tree, "H")
    print(f"  capture done: {len(res)} keys in {time.time() - t0:.0f}s", file=sys.stderr)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--eq-out", default=os.environ.get("EQUIV_OUT"))
    a = ap.parse_args()
    assert a.eq_out, "set EQUIV_OUT (outside the repo)"
    res = capture(os.path.abspath(a.src), a.out, a.fixtures, a.eq_out)
    with open(os.path.join(a.out, "golden.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=0, sort_keys=True)


if __name__ == "__main__":
    main()
