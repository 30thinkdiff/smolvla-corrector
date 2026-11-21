# Source attribution and licenses

This repository is an independent reproduction and empirical study, not the official VLA-Corrector project.

Original research:
- VLA-Corrector: https://github.com/ZJU-OmniAI/vla-corrector (source pinned at 927a06da8f00d7df483459e9039219e076c846de)
- LeRobot: https://github.com/huggingface/lerobot (research environment pinned at 7e241bd630a3719a56157a497ce5d08f244784f1)
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO (research environment pinned at 8f1084e3132a39270c9a13ebe37270a43ece2a01)
- SmolVLA checkpoint: https://huggingface.co/lerobot/smolvla_libero (revision a24f1f380461b665158d52acd140152c79e8b36d)

The **implementation/** directory contains research integration and experiment scripts from the owner's pre-existing `30thinkdiff/corrector` repository at a pinned revision. Files remain byte-for-byte copies except `corrector_online_runtime.py`, which was adapted solely to drop the local exercise-file fallback import. Both the original and published Git blob SHAs are preserved in `provenance/source_map.json`. The files extend and interface with the above third-party projects; no claim is made that all underlying algorithms or imported functions are original contributions. The public project retains the upstream Apache-2.0 license text and cites its source. Copyright notices embedded in copied files remain intact.

The **results/** directory contains a selected set of existing evaluation outputs, not generated for this new portfolio repository. The source paths, immutable commits, and Git blob hashes for every migrated file are stored in provenance/source_map.json.

Model weights, source datasets, original third-party packages, private simulation assets and archived videos are not redistributed. Their separate licensing and model-card restrictions still apply. Please keep this attribution and the relevant upstream notices with any further redistribution.

This file describes attribution, not legal advice or a claim that every dependency uses the same license.
