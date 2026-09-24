"""CLI surface (plan §8.3): --help of all 8 console scripts, config_table rows, import side effects
(K1-K3), pickle probe (K10/K11). `python equiv/cli_surface.py --src <tree> --out <json> --tree <synth>`

Every probe runs in a fresh subprocess through _boot (so `nanounet` is EQUIV_SRC). segtrack gets a
stub `tracking` package (L15: its --help is unreachable without it). The synth tree from the last
capture provides valid argv for the config_table probes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLANS = "nnUNetResEncUNetLPlans"
CLIS = ("preprocess", "train", "pretrain", "predict", "lesion_weights", "build_splits", "build_valset", "segtrack")

STUB = {
    "__init__.py": "",
    "decode.py": "DECODE_CHOICES = ('hungarian', 'greedy')\n",
    "common.py": "from pathlib import Path\nDEPLOYED_CKPT = Path('/nonexistent/track.ckpt')\nDEPLOYED_DUST_TAU = 0.05\n",
    "infer.py": (
        "from types import SimpleNamespace\n"
        "def load_matcher(ckpt, device):\n    return SimpleNamespace(hparams=SimpleNamespace(k_intra=8))\n"
        "def graph_cfg_from_ckpt(m, k):\n    return SimpleNamespace(intra=k, drop_dp=0.0)\n"
    ),
}

PROBE_HELP = """
import sys, io, contextlib
sys.path.insert(0, {here!r}); import _boot
import importlib
m = importlib.import_module('nanounet.cli.' + {cli!r})
sys.argv = ['nanounet_' + {cli!r}, '--help']
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        m.main()
except SystemExit:
    pass
print(buf.getvalue())
"""

PROBE_ROWS = """
import sys, json
sys.path.insert(0, {here!r}); import _boot
import nanounet.common as c
rows = []
class _Stop(BaseException):
    pass
def rec(r, title='config'):
    rows.extend([[str(a), str(b), str(s)] for a, b, s in r]); rows.append(['<title>', title, ''])
    raise _Stop()
c.config_table = rec
import importlib
m = importlib.import_module('nanounet.cli.' + {cli!r})
sys.argv = ['x'] + {argv!r}
try:
    m.main()
except _Stop:
    pass
print('ROWS=' + json.dumps(rows))
"""

PROBE_IMPORT = """
import sys, os, json, warnings, logging
sys.path.insert(0, {here!r}); import _boot
seen = {{}}
class Rec:
    def find_spec(self, name, path=None, target=None):
        if name in ('torch', 'pytorch_lightning') and name not in seen:
            seen[name] = {{'TMPDIR': os.environ.get('TMPDIR'),
                          'n_filters': len(warnings.filters),
                          'leafspec_filter': any('LeafSpec' in str(f[1]) for f in warnings.filters)}}
        return None
sys.meta_path.insert(0, Rec())
import importlib, tempfile
importlib.import_module('nanounet.cli.' + {cli!r})
import torch.multiprocessing as tmp
out = {{'at_import': seen, 'TMPDIR': os.environ.get('TMPDIR'), 'tempdir': tempfile.tempdir,
        'sharing': tmp.get_sharing_strategy(), 'filters': [repr(f) for f in warnings.filters],
        'loggers': {{n: logging.getLogger(n).level for n in ('nanounet', 'pytorch_lightning', 'lightning', 'lightning.pytorch')}},
        'pl_loaded': 'pytorch_lightning' in sys.modules}}
print('IMPORT=' + json.dumps(out, sort_keys=True))
"""

PROBE_PICKLE = """
import sys, pickle, json, functools, importlib
sys.path.insert(0, {here!r}); import _boot
res = {{}}
def chk(name, obj):
    try:
        b = pickle.dumps(obj)
        pickle.loads(b)
        mod = getattr(obj, '__module__', None) or getattr(getattr(obj, 'func', None), '__module__', None) or type(obj).__module__
        importlib.import_module(mod)
        res[name] = 'ok'
        print('MOD', name, mod, file=sys.stderr)
    except Exception as e:
        res[name] = 'FAIL ' + type(e).__name__ + ': ' + str(e)[:200]
from nanounet.train.patch_iterable import worker_init
from nanounet.pretrain import dataset as pd
from nanounet.data.cohorts import CohortSampler
from nanounet.data.blosc2_dataset import Blosc2Folder
from nanounet.data import valset
from nanounet.plan.prep import preprocess, fingerprint
from nanounet.prompt import centroids
chk('worker_init', worker_init)
chk('_worker_init_partial', functools.partial(pd._worker_init, out_dir='.'))
chk('CohortSampler', CohortSampler)
chk('Blosc2Folder', Blosc2Folder)
chk('ValManifest', valset.ValManifest)
chk('preprocess._worker', preprocess._worker)
chk('fingerprint._analyze_case', fingerprint._analyze_case)
chk('centroids._write_centroids_for_case', centroids._write_centroids_for_case)
print('PICKLE=' + json.dumps(res, sort_keys=True))
"""


def _py(code: str, env: dict, cwd: str) -> tuple[str, str]:
    p = subprocess.run([sys.executable, "-c", code], env=env, cwd=cwd, capture_output=True, text=True)
    return p.stdout, p.stderr


def _grab(tag: str, out: str, err: str, what: str):
    for line in out.splitlines():
        if line.startswith(tag + "="):
            return json.loads(line[len(tag) + 1:])
    return {"ERROR": f"{what}: no {tag} line", "stderr": err.strip().splitlines()[-3:]}


def surface(src: str, tree: str, eq_out: str) -> dict:
    stubs = os.path.join(eq_out, "stubs")
    os.makedirs(os.path.join(stubs, "tracking"), exist_ok=True)
    for f, body in STUB.items():
        open(os.path.join(stubs, "tracking", f), "w", encoding="utf-8").write(body)
    env = dict(os.environ, EQUIV_SRC=src, PYTHONPATH=stubs, COLUMNS="100", CUDA_VISIBLE_DEVICES="",
               NANOUNET_RAW=f"{tree}/raw", NANOUNET_PREPROCESSED=f"{tree}/preprocessed",
               NANOUNET_RESULTS=f"{tree}/results", NANOUNET_TMPDIR="/tmp/claude-501/nq", PYTHONHASHSEED="0")
    env.pop("WANDB_RUN_ID", None)
    cwd = tree
    res: dict = {}
    for cli in CLIS:
        out, err = _py(PROBE_HELP.format(here=HERE, cli=cli), env, cwd)
        res[f"help/{cli}"] = out if out.strip() else "ERROR " + err[-500:]
        out, err = _py(PROBE_IMPORT.format(here=HERE, cli=cli), env, cwd)
        res[f"import/{cli}"] = _grab("IMPORT", out, err, cli)
    x = f"{tree}/extra"
    r = f"{tree}/results"
    empty = f"{eq_out}/empty_dir"
    os.makedirs(empty, exist_ok=True)
    open(f"{eq_out}/dummy_track.ckpt", "w").close()
    argvs = {
        "train": ["-d", "903", "--plans", PLANS, "--config", f"{x}/configs/default.json", "--accelerator", "cpu",
                  "--no-wandb", "--out", f"{r}/rows_train"],
        "predict": ["-i", f"{x}/predin", "-o", f"{r}/rows_pred", "-m", f"{r}/F_sup", "--device", "cpu"],
        "segtrack": ["--bl-dir", empty, "--fu-dir", empty, "-m", f"{r}/F_sup", "--track-ckpt",
                     f"{eq_out}/dummy_track.ckpt", "--device", "cpu", "-o", f"{r}/rows_seg"],
    }
    for cli, argv in argvs.items():
        out, err = _py(PROBE_ROWS.format(here=HERE, cli=cli, argv=argv), env, cwd)
        res[f"rows/{cli}"] = _grab("ROWS", out, err, cli)
    out, err = _py(PROBE_PICKLE.format(here=HERE), env, cwd)
    res["pickle"] = _grab("PICKLE", out, err, "pickle")
    res["_pickle_modules"] = [ln for ln in err.splitlines() if ln.startswith("MOD ")]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eq-out", default=os.environ.get("EQUIV_OUT"))
    a = ap.parse_args()
    res = surface(os.path.abspath(a.src), a.tree, a.eq_out)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
