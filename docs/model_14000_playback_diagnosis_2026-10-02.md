# model_14000.pt playback reset diagnosis

The same checkpoint was replayed on the local RTX 4060 with four environments,
seed 1, headless, for 3000 control steps. No training was performed.

Checkpoint: `logs/lite3_pie/Oct02_02-09-57_/model_14000.pt`.

| Playback | Failure terminations | Timeouts | Mean completed episode steps |
| --- | ---: | ---: | ---: |
| Original legacy code at d0ab3c0 | 0 | 12 | 1000 |
| Formal-training source before playback fix | 1246 | 0 | 9.616 |
| Formal-training source after playback fix | 0 | 12 | 1000 |

Every failure before the fix was a torso contact. The old checkpoint was trained
with action clipping at 4, raw command observations, and initial terrain levels
0–1. The formal-training defaults use clipping at 100, command scales
`[2, 2, 0.25]`, and initial levels 0–5. The earlier play entrypoint restored model
weights but created its environment from those newer defaults.

The play entrypoint now restores the saved environment before creating the
simulator. For version-2 v1.0.2 PIE checkpoints, the PPO `schedule` field
distinguishes the historical minimal and formal observation conventions. Newly
saved checkpoints include explicit playback observation scales, observation
clipping, and positive-reward clipping. Runtime environment count, device, and
headless mode remain controlled by the playback invocation. Training defaults
and termination thresholds were not changed by this fix.

The environment reports the condition and pre-reset episode length. Playback
prints the first 20 reset events and counts every condition in its final result.
Multiple simultaneous conditions can be counted for one episode.

After the fix, all resets occur at 1000 steps, i.e. 20 simulated seconds. The
viewer has no real-time pacing, so the wall-clock interval can be shorter.
The robots remain near their starting positions, with base heights around
0.12–0.14 m; correcting playback compatibility does not establish successful
walking or paper-level performance for this checkpoint.

Validation: 11 CPU checks passed for playback configuration and checkpointing,
followed by the four-environment GPU playback reported above. Interactive viewer
behavior and large-scale training were not tested.
