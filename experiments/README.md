# Experiment runners

The experiment is deliberately staged so the test split cannot influence model
selection.

1. `recover_experiment.py` runs coarse or five-fold confirmation candidates on
   the 242-image development pool.
2. `select_candidates.py` validates confirmation evidence and writes an
   immutable configuration lock.
3. `finalize_experiment.py` completes every declared CV/full-pool run before it
   opens the fixed test gate.
4. `analyze_final.py` selects inference policy, seed averaging, blend weights,
   and SVM probes from OOF evidence, writes a second lock, and reports test
   results with bootstrap uncertainty.

`pipeline_source.ipynb` is a code-only, output-free extraction of the audited
training primitives used by the runners. The executed portfolio notebook at the
repository root is intentionally shorter and reads only tracked evidence.
