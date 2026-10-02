# nanochat-style: allow R1 (experiment code, LOC cap waived by owner 2026-09-30)
"""exp08 - External private set: our pipeline against Kirchhoff et al.  (paper: Experiments > Private sets > "A new centre"; Table 'nine experiments' row 8)

QUESTION   Does the system generalise to a new centre it has never seen? The same masks (Dice, NSD, detection) and identity (recall per class,
           edge F1) protocol as exp07, our pipeline against Kirchhoff et al. (LongiSeg), on scan pairs from a partner lab.
WHY        Fills the external-set rows of the paper. The data stays at the partner lab; only the anonymised results.json comes back.
DATA       A folder in the Longitudinal-CT layout from the new centre (`--data-root`, patient list `--patients-csv`); the partner must supply
           the propagated FU points in inputsTrFU/*.json, made with the propagation the public dataset shipped with (the code never registers).
METHOD     Identical to exp07 (this file only sets the experiment name and the site tag; the protocol is `experiments.exp07_internal_set.run.main`).
OUTPUT     Identical to exp07: results.json tables per_patient, per_lesion, metrics, deltas; table.md; artifacts/.
COMMAND    python -m experiments.exp08_external_set.run --data-root /data/site_b --patients-csv /data/site_b/test_patients.csv --tag site_b
DEPENDS ON experiments/exp07_internal_set/run.py (the protocol), kirchhoff.py (same folder), and everything exp07 depends on.
RUNTIME    Same as exp07: about 1-2 h for 60 patients on one A100; resumable; --rescore takes a minute.
CAVEATS    Same as exp07: numbers are not meaningful until the matcher is retrained on the fixed graph cache and `MATCHER_FINAL` is repointed;
           the LongiSeg weights are research-only. Nothing here is tuned on the external data.
"""

from experiments.exp07_internal_set.run import main

EXP = "exp08_external_set"
PAPER = {"section": "Experiments > Private sets > A new centre", "table_row": 8, "supports": "external private set: masks and identity, ours vs Kirchhoff et al."}

if __name__ == "__main__":
    main(EXP, PAPER, __doc__, site="external set, new centre")
