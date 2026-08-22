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
5. `evaluate_faithfulness.py` selects one attribution method per model family
   on a deterministic class-balanced OOF cohort, writes a third lock, and only
   then evaluates perturbation faithfulness and sanity checks on the fixed test
   split.

`pipeline_source.ipynb` is a code-only, output-free extraction of the audited
training primitives used by the runners. The executed portfolio notebook at the
repository root is intentionally shorter and reads only tracked evidence.

Full attribution maps, per-perturbation traces, and checkpoints remain under
the ignored `.runs/` tree. `tools/export_faithfulness_results.py` promotes only
the validated, path-sanitized tables and figures needed for review.
