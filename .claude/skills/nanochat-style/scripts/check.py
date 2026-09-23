#!/usr/bin/env python3
"""nanochat-style checker: the mechanically checkable rules of SKILL.md, run after every edit.

Per-file rules (R1 R2 R3 R4 R6 R11 U8 E1 E4 G2) walk nanounet/**/*.py with `ast`. Cross-file rules
(D3 D4 D6) diff argparse flags and console scripts against docs/, which is how stale docs get
caught. One grep-able line per finding: `path:line: RULE severity message`. `--json` appends one
machine-readable last line. Exit 1 on any `error`. Stdlib only: runs without installing nanounet.

Waive a finding with a comment naming the rule and a reason, on the flagged line or the line above
(file-level rules R1 R2 R6: anywhere in the file):  # nanochat-style: allow R1 (why)
"""

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # <repo>/.claude/skills/nanochat-style/scripts/check.py
PKG = ROOT / "nanounet"
USER_DOCS = ["docs/index.md", "docs/steps", "docs/reference", "README.md"]
MAX_LOC, TINY_LOC, MAX_DOC = 200, 30, 200
BOUNDARY_EXC = {"SystemExit", "FileNotFoundError", "FileExistsError", "ValueError", "RuntimeError", "KeyError"}
HOT_FUNCS = {"forward", "training_step", "compute_loss", "__getitem__", "__iter__", "__next__"}
SYNC_ATTRS = {"item", "cpu", "tolist", "numpy"}
FILE_RULES = {"R1", "R2", "R6"}


def changed_files() -> set[str]:
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    return {p for p in git("diff", "--name-only", "HEAD") + git("ls-files", "--others", "--exclude-standard") if p}


def waived(lines: list[str], rule: str, line: int) -> bool:
    tag = f"nanochat-style: allow {rule}"
    if rule in FILE_RULES:
        return any(tag in l for l in lines)
    return any(tag in lines[i] for i in (line - 1, line - 2) if 0 <= i < len(lines))


def check_py(path: Path, add, flags: dict) -> None:
    rel, src = str(path.relative_to(ROOT)), path.read_text(encoding="utf-8")
    lines, tree = src.splitlines(), ast.parse(src)
    emit = lambda line, rule, sev, msg: None if waived(lines, rule, line) else add(rel, line, rule, sev, msg)
    is_init = path.name == "__init__.py"
    if len(lines) > MAX_LOC:
        emit(1, "R1", "error", f"{len(lines)} LOC > {MAX_LOC}; split on a concept boundary")
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if not is_init and len(lines) < TINY_LOC and len(defs) == 1 and not classes:
        emit(1, "R2", "warn", f"{len(lines)} LOC hosting one function `{defs[0].name}`; inline into a sibling or common.py")
    if not is_init and not ast.get_docstring(tree):
        emit(1, "R6", "error", "missing module docstring (what's inside + non-obvious features)")
    if any(re.search(r"utils?|helpers?", p) for p in path.relative_to(ROOT).parts):
        emit(1, "R4", "error", "`utils`/`helpers` in path; use a noun (geometry.py, centroids.py)")
    for c in classes:
        if re.search(r"(Base|Abstract|Factory|Registry|Mixin|Strategy)", c.name):
            emit(c.lineno, "R3", "warn", f"class `{c.name}` smells like framework ceremony; a function or an `if` instead?")
    hot = {id(f) for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and any(getattr(m, "name", "") == "forward" for m in c.body)
           for f in c.body if isinstance(f, ast.FunctionDef) and not re.match(r"__init__|on_|validation|test|predict", f.name)}
    # ^ every per-step method of a module that has forward(); validation/hooks may sync by design
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(m.split(".")[0] == "abc" for m in mods):
                emit(node.lineno, "R3", "error", "`abc` import; no ABCs")
            if any(m.split(".")[0] == "tqdm" for m in mods):
                emit(node.lineno, "U1", "error", "tqdm; use common.nano_progress")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            if rel != "nanounet/common.py":
                emit(node.lineno, "R11", "error", "bare print(); use cprint / rich renderable (stderr console)")
        elif isinstance(node, ast.ExceptHandler):
            if all(isinstance(s, (ast.Pass, ast.Continue)) for s in node.body):
                broad = node.type is None or ast.unparse(node.type) in ("Exception", "BaseException")
                if broad:
                    emit(node.lineno, "E4", "error", "broad except swallows silently; crash with the fix instead")
                else:
                    emit(node.lineno, "E4", "warn", "narrow swallow: OK only for best-effort side effects, never the result path; waive with reason")
        elif isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
            if node.exc.func.id in BOUNDARY_EXC and "Fix" not in ast.unparse(node.exc):
                emit(node.lineno, "E1", "warn", f"{node.exc.func.id} without 'Fix:' line (what's wrong / expected / what to run)")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (node.name in HOT_FUNCS or id(node) in hot):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr in SYNC_ATTRS:
                    emit(sub.lineno, "G2", "warn", f".{sub.func.attr}() in hot `{node.name}` is a CPU-GPU sync")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            longs = [a.value for a in node.args if isinstance(a, ast.Constant) and str(a.value).startswith("--")]
            for f in longs:
                flags.setdefault(f, (rel, node.lineno))
            kws = {k.arg for k in node.keywords}
            if longs and "help" not in kws:
                emit(node.lineno, "U8", "warn", f"{longs[0]} has no help=")
            if longs and not any("_" not in f for f in longs):
                emit(node.lineno, "U8", "warn", f"{longs[0]} is snake_case; new flags are kebab-case")


def doc_files() -> list[Path]:
    out = []
    for d in USER_DOCS:
        p = ROOT / d
        out += sorted(p.rglob("*.md")) if p.is_dir() else [p] if p.exists() else []
    return out


def check_docs(add, flags: dict) -> None:
    docs = {str(p.relative_to(ROOT)): p.read_text(encoding="utf-8") for p in doc_files()}
    text = "\n".join(docs.values())
    toml = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    scripts = set(re.findall(r"^(nanounet_\w+)\s*=", toml, re.M))
    for rel, body in docs.items():
        n = body.count("\n") + 1
        if rel != "README.md" and n > MAX_DOC:
            add(rel, 1, "D4", "error", f"{n} lines > {MAX_DOC}; split by concept (dev-notes/ and handoffs/ exempt)")
        for i, line in enumerate(body.splitlines(), 1):
            for cmd in set(re.findall(r"(?<![\w/.-])nanounet_[a-z_]+\b(?![/.])", line)) - scripts:
                add(rel, i, "D4", "error", f"`{cmd}` is not a console script in pyproject.toml (stale doc?)")
            if line.startswith("|"):
                for f in set(re.findall(r"`(--[a-z0-9][a-z0-9_-]*)", line)) - set(flags):
                    add(rel, i, "D4", "error", f"documented flag {f} not defined by any CLI (stale doc?)")
    for s in sorted(scripts):
        if s not in text:
            add("pyproject.toml", 1, "D6", "warn", f"console script `{s}` appears in no user doc")
    for f, (rel, line) in sorted(flags.items()):
        if rel.startswith("nanounet/cli/") and f not in text:
            add(rel, line, "D3", "warn", f"{f} missing from every docs argument table")


def main() -> None:
    ap = argparse.ArgumentParser(description="Check nanoUNet against the nanochat-style rules.")
    ap.add_argument("paths", nargs="*", help="Files/dirs to check (default: nanounet/). Repo-relative or absolute.")
    ap.add_argument("--changed", action="store_true", help="Per-file rules only on files changed vs HEAD (+ untracked).")
    ap.add_argument("--no-docs", action="store_true", help="Skip the cross-file docs rules (D3 D4 D6).")
    ap.add_argument("--errors-only", action="store_true", help="Hide warn-level findings.")
    ap.add_argument("--json", action="store_true", help="Append one JSON line {errors, warns, findings} to stdout.")
    args = ap.parse_args()

    findings: list[dict] = []
    add = lambda path, line, rule, sev, msg: findings.append(dict(path=path, line=line, rule=rule, severity=sev, msg=msg))
    roots = [Path(p) if Path(p).is_absolute() else ROOT / p for p in args.paths] or [PKG]
    files = sorted({f for r in roots for f in ([r] if r.is_file() else r.rglob("*.py")) if "__pycache__" not in f.parts})
    if args.changed:
        only = changed_files()
        files = [f for f in files if str(f.relative_to(ROOT)) in only]
    flags: dict = {}
    every = [] if args.no_docs else [f for f in sorted(PKG.rglob("*.py")) if "__pycache__" not in f.parts]
    for f in sorted(set(files) | set(every)):  # unchecked files still contribute their flags to the docs diff
        check_py(f, add if f in files else lambda *a: None, flags)
    if not args.no_docs:
        check_docs(add, flags)

    findings = [dict(t) for t in dict.fromkeys(tuple(f.items()) for f in findings)]  # nested hot defs can double-report
    shown = [f for f in findings if not (args.errors_only and f["severity"] == "warn")]
    for f in sorted(shown, key=lambda f: (f["severity"] != "error", f["path"], f["line"])):
        print(f"{f['path']}:{f['line']}: {f['rule']} {f['severity']} {f['msg']}")
    n_err = sum(f["severity"] == "error" for f in findings)
    n_warn = len(findings) - n_err
    print(f"nanochat-style: {n_err} error(s), {n_warn} warn(s) in {len(files)} file(s)", file=sys.stderr)
    if args.json:
        print(json.dumps(dict(errors=n_err, warns=n_warn, findings=shown)))
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
