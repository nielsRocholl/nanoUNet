"""Shared run record for every paper experiment: paths and constants, environment guards, common flags, run directory.

An experiment's `run.py` parses its arguments, validates all inputs (`problem` + `abort_if`, E6), calls `start_run(...)`,
does the work, and ends with `run.finish(...)`. The run directory is the record: `command.txt` (exact command), `run.json`
(provenance), `log.txt` (console tee), `results.json` (every per-unit number, schema lesionglue-exp/1), one CSV per table,
`table.md`, and `artifacts/` (heavy files, never mirrored). Everything except `artifacts/` is copied into
`experiments/results/<exp>/<run_id>/` so any table or plot can be regenerated from git alone.

Non-obvious choices: `status: running` is written before any work and flipped by an excepthook/atexit, so a crashed run
still has its command and traceback; runs whose tag contains `smoke` mirror into the git-ignored `INDEX_smoke.jsonl`; the
dirty flag only looks at code paths (not results, not unrelated dotfiles); big-file hashes are cached in
`<out-root>/.fingerprints.json`; the mirror copies with `copyfileobj` because `shutil.copy2` fails on the CIFS mount.
"""

# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from core.ui import arg_rows, config_table, console, cprint, nano_header

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = Path("/nnunet_data/experiments")
MIRROR_ROOT = Path(__file__).resolve().parent / "results"
LONGI_ROOT = Path("/nnunet_data/Longitudinal-CT")
HOLDOUT_CSV = LONGI_ROOT / "test_patients.csv"
SEG_MODEL_DIR = Path("/nnunet_data/NanoUNet_results/nanounet/Dataset900_Merged_nnUNetResEncUNetLPlans_h200_smallpv_f0_h200_final_ft250_fromlast")
SEG_CKPT = SEG_MODEL_DIR / "finetune" / "bestsel-epoch=153-val_prompt_score=0.7261.ckpt"  # owner-chosen 2026-09-30, see seg-checkpoint sweep
SEG_EMA = True
MATCHER_FINAL = Path("/nnunet_data/lesion_tracking/runs/final_v9_noval/seed0/last.ckpt")  # selection-free, trained on the 240 train+val pool, graph cache v9 (merge-target nodes, uniGradICON fill)
LONGISEG_DIR = Path("/nnunet_data/LongiSeg")
LONGISEG_MODEL = LONGISEG_DIR / "_model"
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 0
SCHEMA = "lesionglue-exp/1"
MIRROR_MAX_BYTES = 20_000_000  # a mirrored file above this stays on /nnunet_data only (listed in run.json)
CODE_PATHS = ("core", "nanounet", "lesionglue", "segtrack", "experiments", "pyproject.toml", ":(exclude)experiments/results")
NANOUNET_ENV = {  # the image ships wrong values (/nanounet_data), so every run re-exports these
    "NANOUNET_RAW": "/nnunet_data/NanoUNet_raw",
    "NANOUNET_PREPROCESSED": "/nnunet_data/NanoUNet_preprocessed",
    "NANOUNET_RESULTS": "/nnunet_data/NanoUNet_results",
    "NANOUNET_TMPDIR": "/root/.cache/nanounet_tmp",
}
LONGISEG_ENV = {"LongiSeg_raw": "data/raw", "LongiSeg_preprocessed": "data/preprocessed", "LongiSeg_results": "data/results"}
LONGISEG_FIX = "pip3 install --no-deps git+https://github.com/MIC-DKFZ/Longitudinal-Difference-Weighting.git"
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def problem(what: str, expected: str, fix: str) -> str:
    """One E1-shaped startup problem: what is wrong, what was expected, the literal fix."""
    return f"{what}\n     expected: {expected}\n     Fix: {fix}"


def abort_if(problems: list[str]) -> None:
    """E6: report every startup problem at once (exit code 1, no traceback)."""
    if problems:
        raise SystemExit(f"{len(problems)} startup problem(s); fix all, then rerun (each has a Fix: line)\n" + "\n".join(f" {i}. {p}" for i, p in enumerate(problems, 1)))


def missing_paths(paths: dict[str, Path], fix: str) -> list[str]:
    """Problems for every named path that does not exist (feed to abort_if)."""
    return [problem(f"{name} not found: {p}", "an existing path", fix) for name, p in paths.items() if not Path(p).exists()]


def limited(items: list, args: argparse.Namespace) -> list:
    """Apply --limit-patients (-1 = all) to an ordered list of patients/cases."""
    return items if args.limit_patients < 0 else items[: args.limit_patients]


def add_common_args(ap: argparse.ArgumentParser, *, gpu: bool = True, rescore: bool = False) -> None:
    ap.add_argument("--tag", default="run", help="label appended to the run id (letters, digits, . _ -); a tag containing `smoke` keeps the run out of git")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT, help="root of the full run directories (INDEX.jsonl lives here)")
    ap.add_argument("--resume", type=Path, default=None, help="RUN_DIR of an unfinished run: reuse it and skip artifacts that already exist (default: new run)")
    ap.add_argument("--seed", type=int, default=0, help="seed for every random draw in this run (recorded in run.json)")
    ap.add_argument("--limit-patients", type=int, default=-1, help="use only the first N patients/cases; -1 = all (smoke runs)")
    if gpu:
        ap.add_argument("--device", default="cuda", help="torch device for the GPU work")
    if rescore:
        ap.add_argument("--rescore", type=Path, default=None, help="RUN_DIR whose artifacts/ are rescored without prediction; writes a new run dir (default: predict)")


def ensure_nanounet_env() -> dict[str, str]:
    """Point NANOUNET_* at existing dirs (overriding wrong image values, logged); returns what is used."""
    problems = []
    for name, default in NANOUNET_ENV.items():
        cur = os.environ.get(name, "")
        if name == "NANOUNET_TMPDIR":
            Path(cur or default).mkdir(parents=True, exist_ok=True)  # scratch dir, created on demand
        if cur and Path(cur).exists():
            continue
        if Path(default).exists() or name == "NANOUNET_TMPDIR":
            cprint(f"[yellow]env override:[/yellow] {name}={cur or '<unset>'} -> {default}")
            os.environ[name] = default
        else:
            problems.append(problem(f"{name} default {default} does not exist", "the /nnunet_data mount", f"mount /nnunet_data or export {name}=<existing dir>"))
    abort_if(problems)
    return {k: os.environ[k] for k in NANOUNET_ENV}


def ensure_longiseg_env() -> None:
    """Make `import longiseg` work in this Python (LongiSeg source on sys.path) and silence its path warnings."""
    problems = []
    if importlib.util.find_spec("longiseg") is None:
        if (LONGISEG_DIR / "longiseg").is_dir():
            sys.path.insert(0, str(LONGISEG_DIR))
        else:
            problems.append(problem(f"LongiSeg source missing at {LONGISEG_DIR}", "a checkout containing longiseg/", f"git clone https://github.com/MIC-DKFZ/LongiSeg {LONGISEG_DIR}"))
    if importlib.util.find_spec("difference_weighting") is None:
        problems.append(problem("python package difference_weighting is not installed", "difference_weighting 0.1.0 (LongiSeg dependency)", LONGISEG_FIX))
    abort_if(problems)
    for name, sub in LONGISEG_ENV.items():
        if (LONGISEG_DIR / sub).is_dir():
            os.environ.setdefault(name, str(LONGISEG_DIR / sub))


def _git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout


def git_state() -> dict:
    """Commit sha + whether tracked/untracked *code* differs from it (results dirs and dotfiles ignored)."""
    status, diff = _git("status", "--porcelain", "--", *CODE_PATHS), _git("diff", "HEAD", "--", *CODE_PATHS)
    return {"sha": _git("rev-parse", "HEAD").strip(), "dirty": bool(status), "diff_sha256": hashlib.sha256((status + diff).encode()).hexdigest() if status else None}


def fingerprint(path: Path, cache: Path | None = None) -> dict:
    """{path, kind, bytes, mtime_utc, sha256}; files are hashed (cached by path+size+mtime), dirs get a shallow name+size manifest hash."""
    path = Path(path).resolve()
    assert path.exists(), f"fingerprint of missing path {path}"
    st = path.stat()
    out = {"path": str(path), "kind": "dir" if path.is_dir() else "file", "bytes": st.st_size, "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}
    if path.is_dir():
        entries = sorted((p.name, p.stat().st_size) for p in path.iterdir() if p.is_file())
        return {**out, "n_files": len(entries), "sha256": hashlib.sha256(json.dumps(entries).encode()).hexdigest()}
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    known = json.loads(cache.read_text()) if cache is not None and cache.is_file() else {}
    if key not in known:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        known[key] = h.hexdigest()
        if cache is not None:
            cache.write_text(json.dumps(known, indent=1))
    return {**out, "sha256": known[key]}


def clean(o: Any) -> Any:
    """JSON-safe copy: numpy -> python, Path -> str, set/tuple -> list, NaN/inf -> None (strict JSON, no silent NaN)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return [clean(v) for v in sorted(o)]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        return clean(o.item())
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    return str(o) if isinstance(o, Path) else o


def resolved_command(ap: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    """The command with EVERY flag explicit (defaults included), so a run can be repeated after defaults change."""
    parts = ["python", sys.argv[0]]
    for a in ap._actions:  # argparse has no public accessor for the option list
        if not a.option_strings or a.dest == "help":
            continue
        v, flag = getattr(args, a.dest), next(s for s in a.option_strings if s.startswith("--"))
        if isinstance(a, argparse.BooleanOptionalAction):
            parts.append(a.option_strings[0] if v else a.option_strings[1])
        elif isinstance(a, argparse._StoreTrueAction):
            parts += [flag] if v else []
        elif v is not None:
            parts += [flag] + [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
    return shlex.join(parts)


class _LogStream:
    """File-like that feeds the terminal and log.txt; the log shows what a terminal would (ANSI stripped, \\r overwrites the line)."""

    def __init__(self, terminal: Any, path: Path):
        self.terminal, self.fh, self.line = terminal, open(path, "a", encoding="utf-8"), ""
        self.encoding = getattr(terminal, "encoding", "utf-8") or "utf-8"

    def write(self, s: str) -> int:
        self.terminal.write(s)
        for ch in ANSI.sub("", s):
            if ch == "\n":
                self.fh.write(self.line + "\n")
                self.line = ""
            elif ch == "\r":
                self.line = ""
            else:
                self.line += ch
        return len(s)

    def flush(self) -> None:
        self.terminal.flush()
        self.fh.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def close(self) -> None:
        self.fh.write(self.line)
        self.fh.close()


class Run:
    """State of one run across calls: its directory, the run.json record, the console tee and the tables written so far."""

    def __init__(self, exp: str, run_id: str, run_dir: Path, out_root: Path, rec: dict, paper: dict, artifacts: Path):
        self.exp, self.run_id, self.dir, self.out_root, self.rec, self.paper = exp, run_id, run_dir, out_root, rec, paper
        self.artifacts, self.tables, self.smoke, self.t0 = artifacts, {}, "smoke" in run_id.lower(), time.time()
        self.tee = _LogStream(console().file, run_dir / "log.txt")
        console().file = self.tee

    def save(self) -> None:
        (self.dir / "run.json").write_text(json.dumps(clean(self.rec), indent=1))

    def write_table(self, name: str, rows: list[dict]) -> None:
        """Store a tidy table: kept in results.json['tables'] and written as <name>.csv (columns = union of keys, first-seen order)."""
        rows = clean(rows)
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(self.dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else ("" if v is None else v) for k, v in r.items()} for r in rows)
        self.tables[name] = rows

    def _end(self, status: str, **extra: Any) -> None:
        self.rec.update(status=status, finished_utc=_now(), wall_sec=round(time.time() - self.t0, 1), **extra)
        self.save()

    def finish(self, summary: dict, tables: dict[str, list[dict]] | None = None, *, table_md: str = "", definitions: dict | None = None,
               notes: list[str] | None = None, next_cmd: str = "") -> dict:
        """Write results.json + table.md, print the summary panel and `next:`, mirror into git, index, emit the U9 JSON line."""
        for name, rows in (tables or {}).items():
            self.write_table(name, rows)
        base = {"bootstrap": {"unit": "patient", "B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED, "ci": "percentile 95"}}
        results = {"schema": SCHEMA, "exp": self.exp, "run_id": self.run_id, "paper": self.paper, "definitions": {**base, **(definitions or {})},
                   "summary": summary, "tables": self.tables, "notes": notes or []}
        (self.dir / "results.json").write_text(json.dumps(clean(results), indent=None if sum(map(len, self.tables.values())) > 5000 else 1))
        (self.dir / "table.md").write_text(table_md or f"# {self.exp} {self.run_id}\n\n(no table.md supplied; see results.json summary)\n")
        self._end("ok")
        cprint(f"[bold green]done[/bold green] {self.exp} | run: {self.run_id} | wall: {self.rec['wall_sec']} s | dir: {self.dir}")
        for k, v in list(_flat(summary))[:40]:
            cprint(f"  {k}: {v}")
        cprint(f"next: {next_cmd or f'cat {self.dir}/table.md'}")
        self._close()
        mirror = MIRROR_ROOT / self.exp / self.run_id
        files = [p for p in sorted(self.dir.rglob("*")) if p.is_file() and (self.dir / "artifacts") not in p.parents]
        self.rec["mirror_skipped"] = [str(p.relative_to(self.dir)) for p in files if p.stat().st_size > MIRROR_MAX_BYTES]
        self.rec["outputs"]["mirror_dir"] = str(mirror)
        self.save()
        for f in files:
            if str(f.relative_to(self.dir)) not in self.rec["mirror_skipped"]:
                _copy(f, mirror / f.relative_to(self.dir))
        self._index([self.out_root, MIRROR_ROOT], "ok")
        out = {"exp": self.exp, "run_id": self.run_id, "out_dir": str(self.dir), "status": "ok"}
        sys.stdout.write(json.dumps(out) + "\n")
        return out

    def _index(self, roots: list[Path], status: str) -> None:
        line = {"exp": self.exp, "run_id": self.run_id, "command": self.rec["command"], "out_dir": str(self.dir), "mirror_dir": self.rec["outputs"]["mirror_dir"],
                "status": status, "wall_sec": self.rec["wall_sec"], "git_sha": self.rec["git"]["sha"]}
        for root in roots:
            with open(root / ("INDEX_smoke.jsonl" if self.smoke else "INDEX.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")

    def _close(self) -> None:
        if console().file is self.tee:
            console().file = self.tee.terminal
            self.tee.close()


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as a, open(dst, "wb") as b:
        shutil.copyfileobj(a, b)  # not shutil.copy2: utime fails on the CIFS mount


def _flat(d: Any, prefix: str = ""):
    """Leaves of a summary dict as (dotted key, text); a [point, lo, hi] list renders as `p [lo, hi]`."""
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _flat(v, f"{prefix}{k}.")
    elif isinstance(d, (list, tuple)) and len(d) == 3 and all(isinstance(x, (int, float)) for x in d):
        yield prefix[:-1], f"{d[0]:.4g} [{d[1]:.4g}, {d[2]:.4g}]"
    else:
        yield prefix[:-1], f"{d:.4g}" if isinstance(d, float) else str(d)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for key, dist in {"torch": "torch", "lightning": "pytorch-lightning", "torch_geometric": "torch-geometric", "numpy": "numpy", "simpleitk": "SimpleITK"}.items():
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[key] = "not installed"
    return out


def _gpu() -> dict:
    if shutil.which("nvidia-smi") is None:
        return {"name": "none", "mem": None}
    name, mem = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.splitlines()[0].split(", ")
    return {"name": name, "mem": mem}


def start_run(exp: str, ap: argparse.ArgumentParser, args: argparse.Namespace, *, inputs: dict[str, Path] | None = None, paper: dict | None = None) -> Run:
    """Create (or reopen, --resume) the run dir, write command.txt + run.json (status running), start the log tee, print header + config."""
    problems = []
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.tag):
        problems.append(problem(f"--tag {args.tag!r} has unsafe characters", "letters, digits, '.', '_', '-'", "rerun with e.g. --tag paper_v1"))
    rescore = getattr(args, "rescore", None)
    if args.resume is not None and rescore is not None:
        problems.append(problem("--resume and --rescore given together", "one of them", "drop one flag"))
    if args.resume is not None and not (args.resume / "run.json").is_file():
        problems.append(problem(f"--resume {args.resume} is not a run directory", "a dir containing run.json", f"ls {args.out_root}/{exp}/"))
    if rescore is not None and not (rescore / "artifacts").is_dir():
        problems.append(problem(f"--rescore {rescore} has no artifacts/", "a finished predict run dir", f"ls {args.out_root}/{exp}/"))
    abort_if(problems)
    now = datetime.now(timezone.utc)
    rid = args.resume.name if args.resume is not None else f"{now:%Y%m%dT%H%M%SZ}_{args.tag}"
    run_dir = args.resume.resolve() if args.resume is not None else args.out_root / exp / rid
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=args.resume is not None)
    prior = json.loads((run_dir / "run.json").read_text()) if args.resume is not None else {}
    git = git_state()
    typed, resolved = "python " + shlex.join(sys.argv), resolved_command(ap, args)
    with open(run_dir / "command.txt", "a", encoding="utf-8") as f:
        f.write(f"{typed}\n{resolved}\ncwd: {os.getcwd()}\ngit: {git['sha']} dirty={git['dirty']}\n\n")
    rec = {"exp": exp, "run_id": rid, "status": "running", "started_utc": prior.get("started_utc", now.isoformat(timespec="seconds")), "finished_utc": None, "wall_sec": None,
           "command": typed, "argv": sys.argv, "cwd": os.getcwd(), "resolved_args": clean(vars(args)), "resolved_command": resolved, "git": git, "host": socket.gethostname(),
           "gpu": _gpu(), "versions": _versions(), "seeds": {"seed": args.seed}, "env": {}, "inputs": [], "rescored_from": str(rescore) if rescore is not None else None,
           "resumes": prior.get("resumes", []) + ([{"started_utc": now.isoformat(timespec="seconds"), "argv": sys.argv}] if args.resume is not None else []),
           "outputs": {"run_dir": str(run_dir), "mirror_dir": None, "artifacts_dir": str(run_dir / "artifacts")}}
    run = Run(exp, rid, run_dir, args.out_root, rec, paper or {}, rescore / "artifacts" if rescore is not None else run_dir / "artifacts")
    run.save()

    def _crash(tp: type, val: BaseException, tb: Any) -> None:
        run._end("failed", traceback="".join(traceback.format_exception(tp, val, tb)))
        run._index([run.out_root], "failed")
        run._close()
        sys.__excepthook__(tp, val, tb)

    def _exit() -> None:
        if run.rec["status"] == "running":
            run._end("failed", error="process exited before Run.finish() (SystemExit, Ctrl-C or kill)")
            run._index([run.out_root], "failed")
            run._close()

    sys.excepthook = _crash
    atexit.register(_exit)
    nano_header(f"{exp}  |  run {rid}")
    config_table(arg_rows(ap, args))
    rec["env"] = {**ensure_nanounet_env(), **{k: os.environ.get(k) for k in LONGISEG_ENV}}
    rec["inputs"] = [{"name": n, **fingerprint(p, args.out_root / ".fingerprints.json")} for n, p in (inputs or {}).items()]
    run.save()
    return run
