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

PPO defaults restored: Adam 1e-3, gamma 0.99, GAE lambda 0.95, clip 0.2,
5 epochs, 4 minibatches, adaptive scheduling with desired policy KL 0.01, 24 rollout steps, entropy coefficient 0.01, value/estimation/KL
weights 1, gradient norm limit 1. Advantages use native sample-standard-deviation normalization plus 1e-8;
a single-transition smoke helper uses a finite zero-variance fallback.
One native PPO optimizer trains the entire model, including the PIE estimator.
The actor uses posterior means; the auxiliary successor decoder samples the VAE.
Unreported paper dimensions and hyperparameters remain reproduction choices.

The Gaussian uses v1.0.2's native learnable `std`, initialized to 1.0 for new
training. The exp(-5)/exp(2) exploration bounds have been removed; only a dtype
epsilon positivity safeguard remains. Loaded weights keep their learned std.
The Lite3 task defaults to 4096 environments and 15000 learning iterations.
The runner reads the task's save_interval (500 by default) and preserves numbered
checkpoints at completed iterations 500, 1000, and so on. It also writes
checkpoint.pt at the end of each learn() call. Saves use a temporary file and
atomic replacement. Periodic checkpoints include weights and optimizer state;
v1.0.2 PIE optimizer resume restores Adam, adaptive learning rate and iteration.
Simulation episodes and GRU state restart; max_iterations is a total target in train.py.
The PIE runner calls the native OnPolicyRunner.log() and writes
TensorBoard events alongside metrics.jsonl. Every actual PIE reward term is
accumulated across rollouts, cleared only for finished environments and reported
as Mean episode rew_* / Episode/rew_*, using LeggedRobot's weighted, dt-scaled
episode sum divided by the configured maximum episode duration. No reward
formula or weight is changed for logging. Train/mean_reward is a rolling mean of the last
100 completed episode returns; Train/mean_step_reward is the per-transition
rollout reward. Episode lengths count collected steps across rollout boundaries,
independent of randomized timeout counters. PIE VAE/estimation losses have
separate tags. All encoder, recurrent, head and decoder modules are printed
after the native Actor/Critic initialization output. An optional extra_log_string
adds PIE loss, gradient, learning-rate, step-reward and transition rows to the
same console table, without removing any native reward/statistics rows. Stock
tasks that omit this field retain their existing output. metrics.jsonl also
records episode_rewards using the same per-term averaging as TensorBoard.
Old wrapped actor/critic/log_std weight keys are converted on weight loading;
2.2.4 optimizer moments cannot be reused in the v1.0.2 parameterization. The distribution is a native attribute, so sequence
evaluation calls `policy_distribution()`.

Install editable into the dedicated pie-isaacgym environment:

```bash
python -s -m pip install --no-deps --no-build-isolation -e ./rsl_rl
python -s -m pip install --no-deps --no-build-isolation -e .
```

These commands are run from the parent unitree_rl_gym directory.
This clone retains its own Git repository; save its changes independently of
the parent repository. A clean checkout of the tag alone does not contain PIE.
The migration initially received static inspection only. The 2026-10-02 formal
configuration revision passed CPU checks of recurrent learning, policy KL scheduling,
indexed impulse contracts, atomic checkpoints and optimizer resume. No GPU
simulation or full training was run in that revision.
