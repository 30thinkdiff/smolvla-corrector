# Reproduction guide

## CPU-only checks — available now

From the repository root (Python 3.10 or newer; no additional packages):

~~~bash
python scripts/check_source_integrity.py
python scripts/reproduce_results.py --format markdown
python -m unittest discover -s tests -v
python -m compileall -q implementation scripts tests
~~~

The first command verifies committed files against the provenance manifest. Four Object10 CSV files were normalized from CRLF to LF on import; both the source and committed Git blob digests are recorded, and CSV text equality was checked after newline normalization. The second re-aggregates the Object10, cross-suite and LIBERO-Plus records. The third executes regression tests. These do not load models, download datasets, access GPUs or start simulations.

## Scope of the committed evidence

The Object10 results reproduce already-published aggregated figures and pair comparisons. The cross-suite trial ledger contains 880 completed trials in 220 four-method matched groups. The main cross-suite table uses three complete suites (Spatial, Goal and Long), 40 trials per method per suite. Additional initial states and a fixed ten-task LIBERO-90 subset are separate analyses; neither is treated as an independent extension of the main denominator.

The LIBERO-Plus horizon dataset covers 32 selected perturbation conditions per method with one evaluation seed. Do not compare its percentages directly with Object10 or official LIBERO leaderboards.

## Original frozen training and simulation implementation

The **implementation/** directory preserves an integrated set of predictor training, online LVM/OGG, and simulation scripts from the archived action-conditioned Corrector code. The historical runs used a research-specific environment and external upstream checkouts. **This is not a turnkey package that can reload every archived checkpoint.** In particular, the original MLP runs used the separate historical frozen snapshot **night-code-1e80242789**, whereas the action-conditioned model experiments used **action-code-9274415dd6**. The exact historical snapshots remain in the source repository and must not be swapped silently.

External dependencies to obtain and validate before attempting a GPU run:
- VLA-Corrector source at 927a06da8f00d7df483459e9039219e076c846de
- LeRobot source at 7e241bd630a3719a56157a497ce5d08f244784f1
- LIBERO source at 8f1084e3132a39270c9a13ebe37270a43ece2a01
- SmolVLA checkpoint at a24f1f380461b665158d52acd140152c79e8b36d
- Corrector training data, original feature manifest, processor configuration, checkpoints, and exact run protocols

Relevant entry points: **corrector_data_workbench.py**, **corrector_train.py**, **corrector_online_runtime.py**, **smolvla_baseline_benchmark.py**, and **corrector_closed_loop_eval.py**. They require the relevant original source and data paths and, for evaluation, a working GPU/MuJoCo setup. Do not claim a fresh-clone GPU reproduction until that has actually been tested.

## Pending verification (for project author)

- [ ] Run end-to-end training from a clean clone with the exact dependencies.
- [ ] Verify a saved checkpoint restores using its original model factory and source hash.
- [ ] Publish a minimal environment/install manifest tested on another machine.
- [ ] Provide selected original videos with verified task/init/seed and method IDs.
- [ ] Add executable examples for the final curated public interface.
