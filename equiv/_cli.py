"""Run one nanounet CLI main() in a seeded, pinned process: `python equiv/_cli.py <module> args...`.

Needed because 3 CLIs lack a __main__ guard (K15), the console scripts would import the editable
install rather than EQUIV_SRC, and no CLI seeds the global RNGs (K12: net init, MAE masks)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _boot  # noqa: E402

if __name__ == "__main__":
    import importlib

    mod = sys.argv[1]
    sys.argv = [mod.rsplit(".", 1)[-1]] + sys.argv[2:]
    m = importlib.import_module(mod)  # import first: K1 side effects must precede torch
    _boot.check_src()
    _boot.seed_all(0)
    _boot.pin()
    m.main()
