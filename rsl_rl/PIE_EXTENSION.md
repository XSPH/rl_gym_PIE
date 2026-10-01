# PIE extension of RSL-RL v1.0.2

Official source: https://github.com/leggedrobotics/rsl_rl.git

Baseline tag: `v1.0.2`; commit: `2ad79cf0caa85b91721abfe358105f869a784121`.
Selected according to the parent Unitree project's `doc/setup_zh.md` and `doc/setup_en.md`.

The new `PIEActorCritic(ActorCritic)`, `PIEPPO(PPO)`,
`PIERolloutStorage(RolloutStorage)` and `PIEOnPolicyRunner(OnPolicyRunner)`
are under their native `modules`, `algorithms`, `storage` and `runners` packages.
No dependency on legged_gym or the separate mjlab project is introduced.

The original runner imports the new class names for its existing factories and
provides a `_reset_env()` hook. Stock tasks keep their original pair-returning
reset. PIE's runner overrides the hook for its observation dictionary.
PIE collection uses `collect()`; replay keeps complete per-environment
sequences and explicit GRU reset masks. Its dictionary observations and label
contract require the PIE runner, rather than the stock PPO rollout loop.
Native storage tensors hold PPO fields, with extra camera/history/label fields.
Timeout values use the actual pre-reset observation, and GAE traces stop at
both terminations and truncations.

Architecture retained from the preceding implementation:

- Proprioception MLP: 450 → 512 → 256 → 128.
- Depth CNN: 2 → 32 → 64 → 128; 4×4 spatial tokens.
- Transformer: 17 tokens, 128 dimensions, 4 heads, 1 layer, FFN 256, dropout 0.
- GRU: input 2176, hidden 128.
- Heads: velocity 3, foot clearance 4, map latent 32, VAE mean/log-variance 16 each.
- Actor: 100 → 512 → 256 → 128 → 12; critic: 235 → 512 → 256 → 128 → 1.
- Successor decoder: 55 → 128 → 128 → 45.
- Heightmap decoder: 32 → 128 → 128 → 187.

PPO defaults retained: Adam 3e-4, gamma 0.99, GAE lambda 0.95, clip 0.2,
2 epochs, 2 minibatches, entropy coefficient 0.01, value/estimation/KL
weights 1, gradient norm limit 1. Advantages retain population-standard-
deviation normalization to support very small batches.
One native PPO optimizer trains the entire model, including the PIE estimator.
The actor uses posterior means; the auxiliary successor decoder samples the VAE.
Unreported paper dimensions and hyperparameters remain reproduction choices.

The action scale parameter is now v1.0.2's native learnable `std`, bounded
between exp(-5) and exp(2), instead of the previous 2.2.4 `log_std`.
This changes optimizer parameterization while preserving initial standard deviation 0.5.
Old wrapped actor/critic/log_std weight keys are converted on loading; optimizer
resume is unsupported. The distribution is a native attribute, so sequence
evaluation calls `policy_distribution()`.

Install editable into the dedicated pie-isaacgym environment:

```bash
python -s -m pip install --no-deps --no-build-isolation -e ./rsl_rl
python -s -m pip install --no-deps --no-build-isolation -e .
```

These commands are run from the parent unitree_rl_gym directory.
This clone retains its own Git repository; save its changes independently of
the parent repository. A clean checkout of the tag alone does not contain PIE.
This migration received static code inspection only; no tests, simulation,
training or checkpoint playback were run.
