# PIE within RSL-RL v1.0.2

The dependency follows Unitree's `doc/setup_zh.md` / `doc/setup_en.md`:
[tag v1.0.2](https://github.com/leggedrobotics/rsl_rl/tree/v1.0.2),
commit `2ad79cf0caa85b91721abfe358105f869a784121`.
This directory is vendored in the parent PIE repository.

## Native lifecycle

`PIEOnPolicyRunner` inherits `OnPolicyRunner.learn()` directly. Collection uses
the original `act → env.step → process_env_step → compute_returns → update`
loop and the native five-element environment return value. Hooks prepare
multimodal actor inputs, choose `torch.no_grad()` collection, append metrics, and
save schema-4 checkpoints. No separate PIE collection or learning loop exists.
Non-PIE tasks continue to use tensor observations and inference-mode collection.

`PIEPPO` inherits `PPO.update()` directly. The shared implementation owns policy
clipping, value clipping, entropy, diagonal-Gaussian policy KL, adaptive learning
rate, Adam, and gradient clipping. PIE supplies complete recurrent environment
trajectories and an auxiliary-loss hook. Actor gradients and supervised gradients
both reach the estimator; one joint optimizer trains all network components.

`PIERolloutStorage` extends the original transition tensors and inherits its GAE
calculation. Additional side buffers store proprioceptive history, camera frame
indices, current-state supervision, pre-reset successor observations, and reset
masks. The actor has recurrent state; the critic has none.

For timeouts, PIE adds `gamma * V(actual_terminal_critic)` to the reward exactly
once, then native GAE stops its trace at the reset. True failures do not bootstrap.
No reset observation replaces the successor reconstruction target.

## Model and configuration

The task passes native `runner`, `policy`, and `algorithm` configuration dictionaries.
Native policy fields `init_noise_std`, `actor_hidden_dims`, `critic_hidden_dims`,
and `activation` are authoritative and resolved into the effective model config.
Additional encoder/head settings live in `policy.model_config`.
Algorithm parameters use the native `algorithm` dictionary. `ModelConfig`
describes the additional network structure. Training and replay enter through
the registered task and the original Gym scripts.

Retained default architecture:

- Proprioception MLP: 450 → 512 → 256 → 128.
- Two-frame depth CNN: 2 → 32 → 64 → 128; a 4×4 token grid.
- Transformer: 17 tokens, width 128, 4 heads, 1 layer, FFN 256, dropout 0.
- GRUCell: input 2176, hidden 128.
- Heads: velocity 3, foot clearance 4, map latent 32, VAE mean/log-variance 16 each.
- Actor: 100 → 512 → 256 → 128 → 12; critic: 235 → 512 → 256 → 128 → 1.
- Successor decoder: 55 → 128 → 128 → 45; height decoder: 32 → 128 → 128 → 187.

The actor uses posterior means for repeatable PPO action probabilities; only the
successor decoder samples the VAE. The paper does not specify this replay detail
or all hidden widths, so these remain explicit reproduction choices.

Native ActorCritic APIs are available: `act`, `evaluate`, `reset`, and
`get_hidden_states`. `evaluate` computes the feedforward critic value.
`evaluate_actions` performs explicit recurrent training replay. GRU state resets
only for finished environments and is refreshed under new weights after updating.

Formal task settings remain 4096 environments, 24 rollout steps, 5 epochs,
4 minibatches, Adam 1e-3, adaptive schedule, desired KL .01, initial std 1,
gamma .99, lambda .95, clip .2, entropy .01, value/estimation/VAE-KL coefficients 1,
and maximum gradient norm 1. Model std has only a numerical positivity safeguard.

## Visual storage and gradients

FP32 images are stored once per captured frame in a rollout pool. Histories
reference pool indices. A two-frame CNN feature cache reuses unchanged histories
within one logical minibatch; repeated uses retain their full CNN gradient.
Each optimizer step gets a new cache. Non-reentrant CNN activation checkpointing
is retained. There is no image quantization, frozen rollout-feature replacement,
or gradient accumulation.

## Logging, checkpoints, and validation

The original console/TensorBoard reward logging stays in the native runner.
PIE appends all five auxiliary losses, policy KL, gradient norm, learning rate,
step reward, terrain levels, reset counts, and visual-memory/reuse statistics.
The same iteration metrics are written to `metrics.jsonl`.

PIE iteration numbers count completed updates. Periodic `model_500.pt`,
`model_1000.pt`, etc. are saved every 500 updates, and `checkpoint.pt` is saved at
the end of a learn call. Atomic replacement protects an existing checkpoint
against an interrupted write. Schema 4 stores native model/optimizer keys,
effective model config and policy activation, current learning rate, unified
native training/environment configs (NumPy values normalized
to ordinary Python scalars/lists for safe weights-only loading), RNG states,
completed iteration, and accumulated time/steps. Loading earlier formats is
explicitly rejected. Playback can construct a one-environment runner while retaining
the saved four-minibatch training settings; actual training checks that its env
count can supply every minibatch. Resuming restores optimizer/LR/counters; simulation episodes
and GRU state restart, as in native RSL-RL resume.

CPU checks exercise native runner ordering, default stock PPO behavior,
actor privacy, timeout bootstrap, recurrent resets/replay, joint gradients,
dense/indexed visual equivalence, clipping/scheduling, logs, atomic checkpoints,
and schema-4 resume. GPU simulation, 4096-environment memory use, throughput, and
training convergence require separate GPU validation and are not claimed here.

Install editable into the separate native Conda environment from the parent root:

```bash
python -s -m pip install --no-deps --no-build-isolation -e ./rsl_rl
python -s -m pip install --no-deps --no-build-isolation -e .
```
