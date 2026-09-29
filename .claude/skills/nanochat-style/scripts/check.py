#!/usr/bin/env python3
"""nanochat-style checker: the mechanically checkable rules of SKILL.md, run after every edit.

Per-file rules (R1 R2 R3 R4 R6 R11 R21 U8 E1 E4 G2) walk every project's **/*.py with `ast`. R20 walks
each project's package tree. Cross-file rules (D3 D4 D6) diff argparse flags and console scripts against each
project's README.md + docs/, which is how stale docs get caught. One grep-able line per finding: `path:line: RULE severity message`. `--json` appends one
machine-readable last line. Exit 1 on any `error`. Stdlib only: runs without installing anything.

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
PROJECTS = {  # R21: project -> projects it may import. One-way and acyclic; core imports none.
    "core": set(),
    "nanounet": {"core"},
    "lesionglue": {"core"},
    "segtrack": {"core", "nanounet", "lesionglue"},
}
PKGS = [ROOT / p for p in PROJECTS]
CMD_RE = "|".join(p for p in PROJECTS if p != "core")  # console scripts are <project>_<cmd>
SCRATCH = ("dev-notes", "handoffs")  # dated scratch: exempt from D4, must open with Date:/Status: (K9)
MAX_LOC, TINY_LOC, MAX_DOC = 200, 30, 200
BOUNDARY_EXC = {
    "SystemExit", "FileNotFoundError", "FileExistsError", "ValueError", "RuntimeError", "KeyError",
    "TypeError", "ImportError", "NotImplementedError", "OSError",
}
STEP_DOC = {  # D3: <project>/cli/<cmd>.py is documented in <project>/docs/steps/<cmd>.md unless listed here
    "nanounet/cli/train_parser.py": "nanounet/docs/steps/train.md",
    "nanounet/cli/build_valset.py": "nanounet/docs/steps/valset.md",
    "nanounet/cli/build_splits.py": "nanounet/docs/steps/valset.md",
    "segtrack/cli/run.py": "segtrack/README.md",
}
HOT_FUNCS = {"forward", "training_step", "compute_loss", "__getitem__", "__iter__", "__next__"}
SYNC_ATTRS = {"item", "cpu", "tolist", "numpy"}
FILE_RULES = {"R1", "R2", "R6"}
FLAT_OK = {f"{p}/cli" for p in PROJECTS}  # R20: one file per console script, 1:1 with pyproject [project.scripts]
MAX_FLAT, MIN_SUB, MAX_SUB = 6, 2, 8


def changed_files() -> set[str]:
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    return {p for p in git("diff", "--name-only", "HEAD") + git("ls-files", "--others", "--exclude-standard") if p}


def waived(lines: list[str], rule: str, line: int) -> bool:
    tag = f"nanochat-style: allow {rule}"
    if rule in FILE_RULES:
        return any(tag in l for l in lines)
    return any(tag in lines[i] for i in (line - 1, line - 2) if 0 <= i < len(lines))


def _enclosing_fn(parents: dict, node: ast.AST):
    p = parents.get(id(node))
    while p is not None and not isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
        p = parents.get(id(p))
    return p


def _assigns_fix(fn, name: str) -> bool:
    if fn is None:
        return False
    for st in ast.walk(fn):
        tgt, val = None, None
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            tgt, val = st.targets[0], st.value
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
            tgt, val = st.target, st.value
        if isinstance(tgt, ast.Name) and tgt.id == name and isinstance(val, ast.Constant) and isinstance(val.value, str):
            if "Fix" in val.value:
                return True
    return False


def _reraises(body: list) -> bool:
    return any(isinstance(n, ast.Raise) for s in body for n in ast.walk(s))


def _broad(h: ast.ExceptHandler) -> bool:
    return h.type is None or ast.unparse(h.type) in ("Exception", "BaseException")


def check_py(path: Path, add, flags: dict) -> None:
    rel, src = str(path.relative_to(ROOT)), path.read_text(encoding="utf-8")
    proj, sub = rel.split("/")[0], rel.split("/")[1] if rel.count("/") else ""
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
    parents = {id(c): n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(m.split(".")[0] == "abc" for m in mods):
                emit(node.lineno, "R3", "error", "`abc` import; no ABCs")
            if any(m.split(".")[0] == "tqdm" for m in mods):
                emit(node.lineno, "U1", "error", "tqdm; use core.ui.nano_progress")
            top = {m.split(".")[0] for m in mods} if getattr(node, "level", 0) == 0 else set()
            for t in sorted(top & PROJECTS.keys() - {proj} - PROJECTS.get(proj, set())):
                emit(node.lineno, "R21", "error", f"{proj} imports {t}, not a declared dependency; hook up through files on disk "
                     f"or a pipeline project, or declare the edge in PROJECTS (check.py) + the root README")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            if rel != "core/ui.py":
                emit(node.lineno, "R11", "error", "bare print(); use cprint / rich renderable (stderr console)")
        elif isinstance(node, ast.ExceptHandler):
            silent = all(isinstance(s, (ast.Pass, ast.Continue)) for s in node.body)
            if _broad(node) and not _reraises(node.body):
                if silent:
                    emit(node.lineno, "E4", "error", "broad except swallows silently; crash with the fix instead")
                else:
                    emit(node.lineno, "E4", "warn", "broad except does not re-raise")
            elif silent:
                emit(node.lineno, "E4", "warn", "narrow swallow: OK only for best-effort side effects, never the result path; waive with reason")
        elif isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
            if node.exc.func.id in BOUNDARY_EXC and "Fix" not in ast.unparse(node.exc):
                names = [n.id for n in ast.walk(node.exc) if isinstance(n, ast.Name)]
                if not any(_assigns_fix(_enclosing_fn(parents, node), n) for n in names):
                    emit(node.lineno, "E1", "warn", f"{node.exc.func.id} without 'Fix:' line (what's wrong / expected / what to run)")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (node.name in HOT_FUNCS or id(node) in hot):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr in SYNC_ATTRS:
                    emit(sub.lineno, "G2", "warn", f".{sub.func.attr}() in hot `{node.name}` is a CPU-GPU sync")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            longs = [a.value for a in node.args if isinstance(a, ast.Constant) and str(a.value).startswith("--")]
            if any(k.arg == "action" and ast.unparse(k.value).endswith("BooleanOptionalAction") for k in node.keywords):
                longs += ["--no-" + f[2:] for f in longs if f.startswith("--") and not f.startswith("--no-")]
            for f in longs:
                flags.setdefault((f, rel), node.lineno)
            kws = {k.arg for k in node.keywords}
            if longs and "help" not in kws:
                emit(node.lineno, "U8", "warn", f"{longs[0]} has no help=")
            if longs and not any("_" not in f for f in longs):
                emit(node.lineno, "U8", "warn", f"{longs[0]} is snake_case; new flags are kebab-case")
    if sub == "cli" and any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body):
        if not any(isinstance(n, ast.If) and "__name__" in ast.unparse(n.test) and "__main__" in ast.unparse(n.test) for n in tree.body):
            emit(1, "K6", "warn", "main() but no `if __name__ == \"__main__\"` guard")
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        calls = {n.func.id for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        if "nano_header" not in calls or "config_table" not in calls or "next:" not in ast.unparse(main):
            emit(main.lineno, "K7", "warn", "main() must call nano_header, config_table, and emit `next:`")


def check_layout(add) -> None:
    """R20: modules live in concept subfolders, at most <project>/<area>/<concept>/<module>.py (R21: one package per project)."""
    for pkg in PKGS:
        _check_tree(pkg, add)


def _check_tree(pkg: Path, add) -> None:
    dirs = [pkg] + sorted(d for d in pkg.rglob("*") if (d / "__init__.py").is_file() and "__pycache__" not in d.parts)
    for d in dirs:  # only python packages: docs/, configs/, scripts/ are not areas
        rel, depth = str(d.relative_to(ROOT)), len(d.relative_to(pkg).parts)
        mods = sorted(p.stem for p in d.glob("*.py") if p.name != "__init__.py")
        init = d / "__init__.py"
        if depth > 2:
            add(rel, 1, "R20", "error", f"depth {depth} below {pkg.name}/; max is {pkg.name}/<area>/<concept>/")
        if depth == 2 and len(mods) < MIN_SUB:
            add(rel, 1, "R20", "error", f"{len(mods)} module(s); a concept subfolder holds >= {MIN_SUB}, fold it into {d.parent.name}/")
        if depth == 2 and len(mods) > MAX_SUB:
            add(rel, 1, "R20", "warn", f"{len(mods)} modules > {MAX_SUB}; split into two concepts")
        if depth < 2 and rel not in FLAT_OK and len(mods) > MAX_FLAT:
            add(rel, 1, "R20", "warn", f"{len(mods)} flat modules > {MAX_FLAT}; group them into concept subfolders")
        for m in mods:
            if depth and (m == d.name or m.startswith(d.name + "_")):
                add(f"{rel}/{m}.py", 1, "R20", "warn", f"name repeats its folder; {d.name}/{m}.py -> {d.name}/{m.removeprefix(d.name + '_') or '<noun>'}.py")
        if depth and init.is_file() and not ast.get_docstring(ast.parse(init.read_text(encoding="utf-8"))):
            add(str(init.relative_to(ROOT)), 1, "R20", "warn", "subpackage __init__.py needs a one-line docstring naming the concept")


def _module_exists(dotted: str) -> bool:
    parts = dotted.split(".")
    base = ROOT.joinpath(*parts)
    if base.with_suffix(".py").is_file() or (base / "__init__.py").is_file():
        return True
    parent = ROOT.joinpath(*parts[:-1]) if len(parts) > 2 else None
    return parent is not None and parent.with_suffix(".py").is_file()


def doc_files() -> list[Path]:
    """User docs: root README + each project's README.md and docs/ (minus dated scratch)."""
    out = [ROOT / "README.md"]
    for pkg in PKGS:
        out += [pkg / "README.md"] + sorted(p for p in (pkg / "docs").rglob("*.md") if not set(p.parts) & set(SCRATCH))
    return [p for p in out if p.is_file()]


def check_docs(add, flags: dict) -> None:
    docs = {str(p.relative_to(ROOT)): p.read_text(encoding="utf-8") for p in doc_files()}
    text = "\n".join(docs.values())
    toml = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    scripts = set(re.findall(rf"^((?:{CMD_RE})_\w+)\s*=", toml, re.M))
    for rel, body in docs.items():
        n = body.count("\n") + 1
        if not rel.endswith("README.md") and n > MAX_DOC:
            add(rel, 1, "D4", "error", f"{n} lines > {MAX_DOC}; split by concept (dev-notes/ and handoffs/ exempt)")
        for i, line in enumerate(body.splitlines(), 1):
            for cmd in set(re.findall(rf"(?<![\w/.-])(?:{CMD_RE})_[a-z_]+\b(?![/.])", line)) - scripts:
                add(rel, i, "D4", "error", f"`{cmd}` is not a console script in pyproject.toml (stale doc?)")
            if line.startswith("|"):
                for f in set(re.findall(r"`(--[a-z0-9][a-z0-9_-]*)", line)) - {k[0] for k in flags}:
                    add(rel, i, "D4", "error", f"documented flag {f} not defined by any CLI (stale doc?)")
            for mod in set(re.findall(rf"(?<![\w/])(?:{CMD_RE})\.[a-z_.]+", line)):
                if not _module_exists(mod.rstrip(".")):
                    add(rel, i, "K8", "warn", f"`{mod}` does not import")
    for s in sorted(scripts):
        if s not in text:
            add("pyproject.toml", 1, "D6", "warn", f"console script `{s}` appears in no user doc")
    for (flag, rel), line in sorted(flags.items()):
        proj, sub, name = (rel.split("/") + ["", ""])[:3]
        if sub != "cli":
            continue
        step = STEP_DOC.get(rel, f"{proj}/docs/steps/{name.removesuffix('.py')}.md")
        if step not in docs:
            add(rel, 1, "D3", "warn", f"no step doc {step} for this CLI (D2 template, or map it in STEP_DOC)")
        elif flag not in docs[step]:
            add(rel, line, "D3", "warn", f"{flag} missing from {step}")
    for sh in sorted(sh for pkg in PKGS for sh in (pkg / "scripts").glob("*.sh")):
        for i, line in enumerate(sh.read_text(encoding="utf-8").splitlines(), 1):
            for mod in set(re.findall(rf"(?<![\w/])(?:{CMD_RE})\.[a-z_.]+", line)):
                if not _module_exists(mod.rstrip(".")):
                    add(str(sh.relative_to(ROOT)), i, "K8", "warn", f"`{mod}` does not import")
    for folder in [ROOT / "docs" / f for f in SCRATCH] + [pkg / "docs" / f for pkg in PKGS for f in SCRATCH]:
        for p in sorted(folder.glob("*.md")):
            head = "\n".join(p.read_text(encoding="utf-8").splitlines()[:12])
            if not re.search(r"(?m)^Date:", head) or not re.search(r"(?m)^Status:", head):
                add(str(p.relative_to(ROOT)), 1, "K9", "warn", "dev-note/handoff must open with Date: and Status:")


def main() -> None:
    ap = argparse.ArgumentParser(description="Check every project against the nanochat-style rules.")
    ap.add_argument("paths", nargs="*", help="Files/dirs to check (default: every project in PROJECTS). Repo-relative or absolute.")
    ap.add_argument("--changed", action="store_true", help="Per-file rules only on files changed vs HEAD (+ untracked).")
    ap.add_argument("--no-docs", action="store_true", help="Skip the cross-file docs rules (D3 D4 D6).")
    ap.add_argument("--errors-only", action="store_true", help="Hide warn-level findings.")
    ap.add_argument("--json", action="store_true", help="Append one JSON line {errors, warns, findings} to stdout.")
    args = ap.parse_args()

    findings: list[dict] = []
    add = lambda path, line, rule, sev, msg: findings.append(dict(path=path, line=line, rule=rule, severity=sev, msg=msg))
    roots = [Path(p) if Path(p).is_absolute() else ROOT / p for p in args.paths] or PKGS
    files = sorted({f for r in roots for f in ([r] if r.is_file() else r.rglob("*.py")) if "__pycache__" not in f.parts})
    if args.changed:
        only = changed_files()
        files = [f for f in files if str(f.relative_to(ROOT)) in only]
    flags: dict = {}
    every = [] if args.no_docs else [f for pkg in PKGS for f in sorted(pkg.rglob("*.py")) if "__pycache__" not in f.parts]
    for f in sorted(set(files) | set(every)):  # unchecked files still contribute their flags to the docs diff
        check_py(f, add if f in files else lambda *a: None, flags)
    check_layout(add)
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
