"""One gate for the pure-refactor series (plan §8): `python equiv/run.py --base <sha> [--allow-text]`.

1. ast_guard.py: only moves/renames (or, with --allow-text, help/error text) vs <sha>.
2. golden: capture.py at <sha> (cached in $EQUIV_OUT/<sha>/, git worktree) and at the working tree;
   every key must match bit-for-bit. Stage H loads the base run's ckpts (old-ckpt load gate).
3. cli_surface.py: --help / config_table rows / import side effects / pickle probe must match;
   --allow-text lets --help text differ and prints the diff for review.
`--selfcheck` captures the base twice and diffs (PR-0). Exits non-zero on any failure."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _sh(argv, **kw):
    return subprocess.run(argv, cwd=REPO, **kw)


def _capture(src: str, out: str, eq: str) -> dict:
    os.makedirs(out, exist_ok=True)
    p = _sh([sys.executable, os.path.join(HERE, "capture.py"), "--src", src, "--out", out,
             "--fixtures", os.path.join(eq, "fixtures"), "--eq-out", eq])
    if p.returncode:
        raise SystemExit(f"capture failed for {src}")
    p = _sh([sys.executable, os.path.join(HERE, "cli_surface.py"), "--src", src, "--tree", os.path.join(eq, "tree"),
             "--out", os.path.join(out, "surface.json"), "--eq-out", eq])
    if p.returncode:
        raise SystemExit(f"cli_surface failed for {src}")
    return {"golden": json.load(open(os.path.join(out, "golden.json"))),
            "surface": json.load(open(os.path.join(out, "surface.json")))}


def _base(sha: str, eq: str, tag: str = "golden") -> dict:
    out = os.path.join(eq, sha[:8], tag)
    if os.path.isfile(os.path.join(out, "surface.json")):
        return {"golden": json.load(open(os.path.join(out, "golden.json"))),
                "surface": json.load(open(os.path.join(out, "surface.json")))}
    wt = os.path.join(eq, f"wt-{sha[:8]}")
    if not os.path.isdir(wt):
        _sh(["git", "worktree", "add", "-q", "--detach", wt, sha], check=True)
    return _capture(wt, out, eq)


def _diff(a: dict, b: dict, allow_text: bool) -> list[str]:
    bad = []
    ga, gb = a["golden"], b["golden"]
    for k in sorted(set(ga) | set(gb)):
        if ga.get(k) != gb.get(k):
            bad.append(f"GOLDEN {k}: {ga.get(k)} -> {gb.get(k)}")
    sa, sb = a["surface"], b["surface"]
    for k in sorted(set(sa) | set(sb)):
        if k.startswith("_") or sa.get(k) == sb.get(k):
            continue
        if allow_text and k.startswith("help/") and isinstance(sa.get(k), str) and isinstance(sb.get(k), str):
            d = difflib.unified_diff(sa[k].splitlines(), sb[k].splitlines(), f"base:{k}", f"head:{k}", lineterm="", n=0)
            print("\n".join(d), file=sys.stderr)
            continue
        bad.append(f"SURFACE {k}: {json.dumps(sa.get(k))[:300]} -> {json.dumps(sb.get(k))[:300]}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--allow-text", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    eq = os.environ.get("EQUIV_OUT")
    assert eq and not os.path.abspath(eq).startswith(REPO), "set EQUIV_OUT to a dir outside the repo"
    sha = _sh(["git", "rev-parse", a.base], capture_output=True, text=True, check=True).stdout.strip()
    if a.selfcheck:
        r1 = _base(sha, eq, "golden")
        r2 = _base(sha, eq, "selfcheck")
        bad = _diff(r1, r2, False)
        for b in bad:
            print(f"  FAIL  {b}", file=sys.stderr)
        print(f"selfcheck: {len(r1['golden'])} golden keys, {len(r1['surface'])} surface keys, {len(bad)} diffs",
              file=sys.stderr)
        return 1 if bad else 0
    g = _sh([sys.executable, os.path.join(HERE, "ast_guard.py"), "--base", sha] + (["--allow-text"] if a.allow_text else []))
    base = _base(sha, eq)
    head_out = os.path.join(eq, "head")
    shutil.rmtree(head_out, ignore_errors=True)
    head = _capture(REPO, head_out, eq)
    bad = _diff(base, head, a.allow_text)
    for b in bad:
        print(f"  FAIL  {b}", file=sys.stderr)
    print(f"equiv: ast_guard {'ok' if g.returncode == 0 else 'FAIL'} | golden {len(head['golden'])} keys | "
          f"surface {len(head['surface'])} keys | {len(bad)} diffs", file=sys.stderr)
    return 1 if (bad or g.returncode) else 0


if __name__ == "__main__":
    sys.exit(main())
