# Blind PIE no-render regression audit

## Scope and versions

Requested investigation: determine whether skipping camera rendering introduced
the subsequent `non-finite PPO/estimator loss` failures. No new simulation or GPU
training was started, and no training configuration or runtime code was changed.

- Original zero-input experiment: `d7b2dd1`.
- Camera-rendering optimization: `3efd1ac`.
- Subsequent user reward adjustments: `57bde6f`, current local and remote HEAD.
- Remote checkout: `asuka@10.70.205.234:/home/asuka/rl_gym_PIE_native`.

## Verified differences

The downloaded `Oct08_20-48-30_/model_500.pt` records checkpoint version 4,
`camera.input_mode=zero`, camera noise zero, and the same model and training
configuration as the current experiment. Therefore both versions supply zero
depth images to the policy. Turning off rendering did not newly remove policy
visual information.

The rendering commit skips Warp camera construction and render/encode in zero
training, and replaces transient `zeros_like` images with an immutable expanded
zero image. The queue/history buffers still own copied values. Capture counters,
frame IDs, update period, delay, and resets retain their original logic.

There are no changes to model, algorithm, storage, or runner source between
`d7b2dd1` and `57bde6f`. The later configuration commit also enables two rewards:

| Field | model_500 configuration | Current configuration |
| --- | --- | --- |
| camera.render_for_debug | absent; camera rendering always ran | false |
| rewards.scales.torques | 0 | -0.0001 |
| rewards.scales.base_height | 0 | -1 |

The saved environment uses a resolved asset path and contains the runtime-derived
`domain_rand.push_interval`; these are serialization/runtime differences rather
than additional source edits. Training configuration comparison found no changes.

An unsuitable inherited value is now active: `rewards.base_height_target=1.0`
metres, while Lite3's configured initial height is 0.30 metres. The actual reward
function uses squared root height error. At 0.30 metres the newly enabled term
contributes -0.49 before dt scaling. `only_positive_rewards=True` can then clip
the aggregate reward to zero. This affects learning, but does not by itself prove
the cause of the non-finite loss. The reward weights and height target were not
changed during this audit.

## Observed remote logs

- Original run `Oct08_20-48-30_`: 552 logged iterations; model_500 exists;
  peak logged pre-clipping gradient norm 37.322; maximum logged learning rate
  0.003375.
- Run `Oct08_22-03-37_`: 136 logged iterations; peak logged pre-clipping
  gradient norm 1,138,172 at iteration 96; maximum logged learning rate 0.01.
- Latest run `Oct08_22-14-44_`: 99 logged iterations; maximum logged learning
  rate 0.01. The last completed iteration has finite loss and gradients.

The failure is raised before iteration metrics are written, so these files do
not identify which loss or tensor first became non-finite. A short log does not,
by itself, establish whether a run failed or was stopped manually.

## CPU checks performed

Command:

```bash
CUDA_VISIBLE_DEVICES='' /home/asuka/Legged/parkour/.conda-envs/pie-isaacgym-native/bin/python \
  -m pytest tests/test_pie_blind_flat.py tests/test_pie_visual_reuse.py -q
```

Result: **45 passed**, 9 PyTorch autocast deprecation warnings, 12.12 seconds.
These include zero-depth lifecycle, raw-image independence, indexed frame pool,
feature-cache gradient equivalence, and multi-epoch recurrent PPO equivalence.

Additional direct comparison was run using `/tmp/pie_camera_audit.py`. It extracts
the original `_render_depth` function from `d7b2dd1` with Python AST, executes it
with a deterministic CPU camera substitute, and compares it with the current
no-camera implementation. Both tasks use the same seeds, proprioceptive noise,
actuator randomization, and observations.

- 20 lifecycle operations including initialization and two partial resets:
  queue/history values, capture counters, frame IDs, observations, targets,
  actuator randomization, and CPU RNG states all match exactly.
- Actual model_500 weights, current full ModelConfig: policy means, GRU states,
  critic values, auxiliary losses, and all 67 populated parameter gradients
  match exactly and remain finite on the controlled inputs.

## Conclusion and remaining evidence

No buffer, model, or PPO data-path regression was found in the no-render changes
under controlled CPU inputs. This does not rule out GPU timing/interoperability
issues or later numerical instability during real training.

The earlier successful run and newer runs differ in both rendering and reward
configuration. A useful GPU comparison must keep policy input zero and fix the
same rewards, seed, model, and optimizer settings while changing rendering only.
Separately, diagnostics must capture the failed batch's individual losses,
log-probability differences, policy standard deviations, and non-finite gradients
to identify the immediate failure. Changing learning rate first would not isolate
whether removal of rendering is causal.
