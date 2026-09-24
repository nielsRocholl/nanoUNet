"""AST guard (plan §8.1): prove the working tree only moved/renamed code relative to a base SHA.

`python equiv/ast_guard.py --base <sha> [--allow-text]`

Every def/class under nanounet/ is indexed by (module, qualname) at <sha> (git show) and in the
working tree, docstring dropped, `ast.dump(include_attributes=False)` (line numbers ignored;
decorators, defaults, annotations included), with renames.json applied. Verdicts: OK/MOVED pass;
CHANGED, MISSING (unless listed in `deleted`) and NEW (unless listed in `new`) fail. Module-level
non-import statements get the same treatment, plus per-module order. --allow-text (C PRs) blanks
raise/assert/`.error()` message args and drops `help=` kwargs before comparing. The K1 CLI files
must keep an identical statement prefix up to their last module-level call. Contract checks
(§8.1 last bullet) compare frozen names/strings between base and HEAD."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import re
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
K1_FILES = ("nanounet.cli.train", "nanounet.cli.pretrain", "nanounet.cli.segtrack")
FROZEN_STRINGS = (
    # K9 sidecar keys, K8 plans strings, K4 ckpt key, K14 env vars
    "centroids_zyx", "bboxes_zyx", "seed_zyx", "volume_vox", "resample_data_or_seg_to_shape", "SimpleITKIO",
    "EMACallback", "NANOUNET_RAW", "NANOUNET_PREPROCESSED", "NANOUNET_RESULTS", "NANOUNET_DEF_N_PROC",
    "NANOUNET_TMPDIR", "NANOUNET_ALLOW_ROOT_CGROUP", "NANOUNET_DL_KEEP_WORKERS", "NANOUNET_MAE_KEEP_WORKERS",
    "NANOUNET_DL_FORCE_NO_WORKERS", "NANOUNET_MEM_DIAG", "NANOUNET_MEM_LOG_EVERY", "NANOUNET_N_PROC_DA",
    "NANOUNET_SINGLE_PATCH_ACCUM_DTYPE", "NANOUNET_SEGTRACK_MODEL", "NANOUNET_SEGTRACK_TRACK", "WANDB_RUN_ID",
    "SLURM_JOB_ID", "self.net =", "_EMA_CB",
)


def _git_files(sha: str) -> dict[str, str]:
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", sha, "nanounet"], cwd=REPO, check=True,
                           capture_output=True, text=True).stdout.split()
    out = {}
    for n in names:
        if n.endswith(".py"):
            out[n] = subprocess.run(["git", "show", f"{sha}:{n}"], cwd=REPO, check=True, capture_output=True,
                                    text=True).stdout
    return out


def _work_files() -> dict[str, str]:
    out = {}
    for root, _, files in os.walk(os.path.join(REPO, "nanounet")):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                out[os.path.relpath(p, REPO)] = open(p, encoding="utf-8").read()
    return out


def _modname(path: str) -> str:
    m = path[:-3].replace(os.sep, ".").replace("/", ".")
    return m[: -len(".__init__")] if m.endswith(".__init__") else m


class _Norm(ast.NodeTransformer):
    def __init__(self, ren: dict, allow_text: bool):
        self.sym, self.mods, self.allow_text = ren["symbols"], ren["modules"], allow_text

    def visit_Name(self, n):
        n.id = self.sym.get(n.id, n.id)
        return n

    def visit_Attribute(self, n):
        self.generic_visit(n)
        n.attr = self.sym.get(n.attr, n.attr)
        return n

    def visit_alias(self, n):
        n.name = self.mods.get(n.name, self.sym.get(n.name, n.name))
        if n.asname:
            n.asname = self.sym.get(n.asname, n.asname)
        return n

    def visit_ImportFrom(self, n):
        self.generic_visit(n)
        if n.module:
            n.module = self.mods.get(n.module, n.module)
        return n

    def _defname(self, n):
        n.name = self.sym.get(n.name, n.name)
        body = n.body
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            n.body = body[1:] or [ast.Pass()]
        self.generic_visit(n)
        return n

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _defname

    def visit_Raise(self, n):
        if self.allow_text and isinstance(n.exc, ast.Call):
            n.exc.args = [ast.Constant("<text>") for _ in n.exc.args]
            n.exc.keywords = []
        self.generic_visit(n)
        return n

    def visit_Assert(self, n):
        if self.allow_text and n.msg is not None:
            n.msg = ast.Constant("<text>")
        self.generic_visit(n)
        return n

    def visit_Call(self, n):
        self.generic_visit(n)
        if self.allow_text:
            n.keywords = [k for k in n.keywords if k.arg != "help"]
            if isinstance(n.func, ast.Attribute) and n.func.attr == "error":
                n.args = [ast.Constant("<text>") for _ in n.args]
        return n


def _index(files: dict[str, str], ren: dict, allow_text: bool):
    defs: dict[tuple[str, str], str] = {}
    stmts: dict[str, list[str]] = {}
    imports: dict[str, list[str]] = {}
    prefix: dict[str, list[str]] = {}
    for path, src in files.items():
        mod = ren["modules"].get(_modname(path), _modname(path))
        tree = ast.parse(src)
        body = tree.body
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
            body = body[1:]
        norm = _Norm(ren, allow_text)

        def walk(nodes, qual):
            for n in nodes:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    nn = norm.visit(copy.deepcopy(n))
                    q = f"{qual}{nn.name}"
                    defs[(mod, q)] = ast.dump(nn, include_attributes=False)
                    walk(n.body, q + ".")

        walk(body, "")
        dumps = [ast.dump(norm.visit(copy.deepcopy(n)), include_attributes=False) for n in body]
        stmts[mod] = [d for n, d in zip(body, dumps) if not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                                                           ast.AsyncFunctionDef, ast.ClassDef))]
        imports[mod] = [d for n, d in zip(body, dumps) if isinstance(n, (ast.Import, ast.ImportFrom))]
        last_call = max((i for i, n in enumerate(body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)),
                        default=-1)
        prefix[mod] = dumps[: last_call + 1]
    return defs, stmts, imports, prefix


def _assigned(d: str) -> str | None:
    m = re.match(r"Assign\(targets=\[Name\(id='(\w+)'", d) or re.match(r"AnnAssign\(target=Name\(id='(\w+)'", d)
    return m.group(1) if m else None


def _contracts(base: dict[str, str], head: dict[str, str]) -> list[str]:
    bad = []
    bt, ht = "\n".join(base.values()), "\n".join(head.values())
    for s in FROZEN_STRINGS:
        if bt.count(s) != ht.count(s):
            bad.append(f"CONTRACT string {s!r}: {bt.count(s)} occurrences at base vs {ht.count(s)} now")

    def ctor_and_fields(files):
        out = {}
        for path, src in files.items():
            for n in ast.walk(ast.parse(src)):
                if isinstance(n, ast.ClassDef) and n.name in ("NanoUNetLM", "NanoMAELM", "EMACallback"):
                    init = next((f for f in n.body if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
                    out[n.name] = [a.arg for a in init.args.args] if init else []
                if path.endswith("nanounet/config.py") and isinstance(n, ast.ClassDef):
                    out["config." + n.name] = [s.target.id for s in n.body if isinstance(s, ast.AnnAssign)]
        return out

    cb, ch = ctor_and_fields(base), ctor_and_fields(head)
    if sorted(cb.values()) != sorted(ch.values()):
        bad.append(f"CONTRACT ctor kwargs / config fields changed: {cb} vs {ch}")
    return bad


def _sh_imports() -> list[str]:
    bad = []
    for root, _, files in os.walk(os.path.join(REPO, "scripts")):
        for f in files:
            if not f.endswith(".sh"):
                continue
            txt = open(os.path.join(root, f), encoding="utf-8").read()
            for mod, names in re.findall(r"from (nanounet[\w.]*) import ([\w, ]+)", txt):
                code = f"from {mod} import {names}"
                env = dict(os.environ, EQUIV_SRC=REPO)
                p = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {HERE!r}); import _boot; {code}"],
                                   env=env, capture_output=True, text=True)
                if p.returncode:
                    bad.append(f"CONTRACT {f}: `{code}` fails: {p.stderr.strip().splitlines()[-1]}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--allow-text", action="store_true")
    a = ap.parse_args()
    ren = json.load(open(os.path.join(HERE, "renames.json"), encoding="utf-8"))
    ren_id = {"symbols": {}, "modules": {}}
    base_files, head_files = _git_files(a.base), _work_files()
    bd, bs, bi, bp = _index(base_files, ren, a.allow_text)
    hd, hs, hi, hp = _index(head_files, ren_id, a.allow_text)
    deleted, new = set(ren.get("deleted", [])), set(ren.get("new", []))
    fails, notes = [], []
    head_by_body: dict[str, list] = {}
    for k, v in hd.items():
        head_by_body.setdefault(v, []).append(k)
    for (mod, q), v in sorted(bd.items()):
        if hd.get((mod, q)) == v:
            continue
        if f"{mod}:{q}" in deleted:
            notes.append(f"DELETED {mod}:{q}")
            continue
        cands = [k for k in head_by_body.get(v, []) if k[1] == q]
        if cands:
            notes.append(f"MOVED {mod}:{q} -> {cands[0][0]}")
            continue
        fails.append(f"{'CHANGED' if (mod, q) in hd else 'MISSING'} {mod}:{q}")
    base_bodies = set(bd.values())
    for (mod, q), v in sorted(hd.items()):
        if (mod, q) not in bd and v not in base_bodies and f"{mod}:{q}" not in new:
            fails.append(f"NEW {mod}:{q}")
    # module-level statements: multiset across the package, plus per-module order
    all_head = Counter(d for L in hs.values() for d in L)
    all_base = Counter(d for L in bs.values() for d in L)
    for d, c in (all_base - all_head).items():
        mods = [m for m, L in bs.items() if d in L]
        if not all(f"{m}:{_assigned(d)}" in deleted for m in mods):
            fails.append(f"STMT removed/changed in {mods}: {d[:160]}")
    for d, c in (all_head - all_base).items():
        mods = [m for m, L in hs.items() if d in L]
        if not all(f"{m}:{_assigned(d)}" in new or f"{m}:<stmt>" in new for m in mods):
            fails.append(f"STMT new in {mods}: {d[:160]}")
    for mod, L in bs.items():
        H = hs.get(mod, [])
        keep = [d for d in L if d in H]
        if keep != [d for d in H if d in keep]:
            fails.append(f"STMT order changed in {mod}")
    for mod in K1_FILES:
        if bp.get(mod) != hp.get(mod):
            fails.append(f"K1 import-order prefix changed in {mod}")
    for mod in sorted(set(bi) | set(hi)):
        if sorted(bi.get(mod, [])) != sorted(hi.get(mod, [])):
            notes.append(f"IMPORTS differ in {mod}")
    fails += _contracts(base_files, head_files) + _sh_imports()
    for n in notes:
        print(f"  note  {n}", file=sys.stderr)
    for f in fails:
        print(f"  FAIL  {f}", file=sys.stderr)
    print(f"ast_guard: {len(bd)} base defs, {len(hd)} head defs, {len(notes)} notes, {len(fails)} failures",
          file=sys.stderr)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
