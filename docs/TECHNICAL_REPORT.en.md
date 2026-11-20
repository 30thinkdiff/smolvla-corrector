# Technical Report — SmolVLA-Corrector

<p align="center">
  <a href="TECHNICAL_REPORT.md">简体中文</a> · <strong>English</strong>
</p>

[English README](../README.en.md) · [中文README](../README.md) · [Reproduction guide](REPRODUCTION.md)

> Working document. The experimental numbers below can be reproduced from committed evidence. Sections marked **AUTHOR TO COMPLETE** require original technical explanations, code walkthroughs, and reviewed qualitative cases. Do not delete those labels before the content has been checked.

## Abstract

**AUTHOR TO COMPLETE (100–150 words).** State the problem, actual reproduced method, experiment setting, one supported positive result, and the central negative cross-suite finding. Avoid generic claims such as “significantly improves robustness” without a specified protocol.

## 1. Problem and research questions

**AUTHOR TO COMPLETE.** Define: why fixed action chunks can accumulate state mismatch; what the Corrector predicts; why event-triggered replanning is more expensive than it first appears; what is being measured.

Suggested questions:
- Does visual dynamics monitoring improve closed-loop success compared with fixed execution horizons?
- How much VLA resampling and monitoring wall time does it require?
- Does lower offline prediction loss correlate with more successful closed-loop behavior?
- Does the Object10 outcome transfer across other LIBERO suites?

## 2. Method

### 2.1 Upstream method and precise attribution

The underlying visual dynamics prediction and LVM/OGG formulation originate from the original VLA-Corrector work (see THIRD_PARTY.md). **AUTHOR TO COMPLETE:** Describe the method by tracing real modules, not by copying an upstream schematic without attribution.

### 2.2 Inputs, prediction target, and training loss

**AUTHOR TO COMPLETE:**
- Observation modalities, encoder tensor shape, action condition, temporal horizon, timestamp units.
- Model-specific differences: MLP, single-/three-frame Transformer, action-conditioned Transformer.
- Exact loss formula, normalization, validation split, learning rate schedule, and inference precision.
- Explicitly distinguish the visual dynamics training loss from the online OGG objective.

### 2.3 Online behavior

**AUTHOR TO COMPLETE:** Explain the monitor state machine, initial calibration, threshold update, score eligibility, queue interruption, and action resampling. Specify when a score is unavailable; do not confuse bootstrap with unscored control steps.

### 2.4 Engineering extensions and scope

**AUTHOR TO COMPLETE:** Clearly separate upstream source, modified integration code, multi-frame/action-sequence experiments, cross-layer diagnostics, additional controller experiments, and replay tooling. Name the actual corresponding source files and immutable run revisions.

## 3. Evaluation protocol

- Policy: pinned SmolVLA-LIBERO checkpoint; upstream environment and policy revisions in THIRD_PARTY.md.
- Object10: 10 manipulation tasks, 454 demonstration episodes, 66,984 observed frames; 15 predictor training runs (5 configurations × 3 training seeds). The main example uses 60 matched trials *per method*.
- Cross-suite: 880 formal trials in 220 four-method groups. The primary comparison is 3 suites × 40 matched trials *per method*; additional initial states and the fixed LIBERO-90 subset have separate denominators.
- LIBERO-Plus horizon: 32 selected conditions per method, one evaluation seed.

**AUTHOR TO COMPLETE:** Add exact control Hz, generated chunk length, executed prefix, environment step limit, timestep/initial-state identity, random seed protocol, checkpoint hashes, model call measurement, and failure/success definition. Confirm the base policy's task-level training coverage before using terms like “zero-shot.”

## 4. Findings

### 4.1 Object10 paired outcomes

| Strategy | Successes / trials |
| --- | ---: |
| Fixed H50 | 44/60 |
| Fixed H20 | 51/60 |
| MLP + LVM | 55/60 |
| MLP + OGG | 55/60 |

The OGG method has **no additional Object10 success-rate gain over LVM** in the recorded 60 matched conditions. Comparing policy chunk calls alone is not a complete runtime comparison; LVM has nonzero monitoring cost.

### 4.2 Cross-suite transfer

| Strategy | Spatial + Goal + Long, primary comparison |
| --- | ---: |
| Fixed H50 | 81/120 |
| Fixed H20 | 93/120 |
| MLP + LVM | 80/120 |
| MLP + OGG | 77/120 |

**AUTHOR TO COMPLETE:** Explain which failure categories differ by suite, how many paired wins/losses each method has, and why the observed difference cannot automatically be attributed to a single cause. Acknowledge that one selected MLP training seed is not evidence of universal generalization.

### 4.3 Monitoring coverage and action conditioning

**AUTHOR TO COMPLETE:** Use the monitor_coverage.csv counts to compare single-action predictors (100% after defined warmup) and ten-action-conditioned models (approximately 59% scored-step coverage). Discuss prediction invalidation, cancellation and action queue timing without implying proven causality for the full success gap.

### 4.4 Execution horizon versus computation

See results/plus_horizon. The 32 selected-condition counts were H50 7/32, H20 20/32, H10 19/32, H5 15/32, H1 6/32. **AUTHOR TO COMPLETE:** Add a verified interpretation of performance / compute tradeoffs, with scope limitations.

### 4.5 Qualitative evidence

**AUTHOR TO COMPLETE:** Insert 2–3 *real, original* paired rollout videos and their exact task, initialization, seed and method IDs. Include at least one negative case. Mark selected case studies as illustrations, not representative random samples.

## 5. Limitations and next hypotheses

**AUTHOR TO COMPLETE.** At minimum cover:
- No stable cross-suite advantage over H20 in the primary comparison.
- OGG gradient updates do not guarantee additional success.
- Online monitoring has nontrivial wall-time overhead.
- Offline supervised prediction metrics do not directly measure task success.
- Monitoring coverage depends on whether the required action condition remains available.
- Repeated initial-state/seed measurements are not automatically independent robot scenes.
- Current artifacts are not a fresh-clone GPU reproducibility guarantee.

Each causal claim should be identified as **measured evidence**, **inference**, or **unverified hypothesis**.

## 6. Conclusion

**AUTHOR TO COMPLETE (80–120 words).** State both positive and negative conclusions and a narrowly motivated next experiment.

## Appendix A. How to reproduce the public tables

~~~bash
python scripts/check_source_integrity.py
python scripts/reproduce_results.py --format markdown
python -m unittest discover -s tests -v
~~~

## Appendix B. Source provenance

See provenance/source_map.json, docs/REPRODUCTION.md and THIRD_PARTY.md.

## Appendix C. Bibliography

- Pan et al. VLA-Corrector: Lightweight Detect-and-Correct Inference for Adaptive Action Horizon. NeurIPS 2026. https://github.com/ZJU-OmniAI/vla-corrector
- LeRobot / SmolVLA source and version: https://github.com/huggingface/lerobot
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO

**AUTHOR TO COMPLETE:** Add exact related work references used in the text; check formal citation details before publication.

---

**Maintenance:** This English report follows the primary [Chinese technical report](TECHNICAL_REPORT.md). Keep results, limitations, methods and references consistent with the pinned research evidence.
