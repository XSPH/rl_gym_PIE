# PIE within RSL-RL v1.0.2

The dependency follows Unitree's `doc/setup_zh.md` / `doc/setup_en.md`:
[tag v1.0.2](https://github.com/leggedrobotics/rsl_rl/tree/v1.0.2),
commit `2ad79cf0caa85b91721abfe358105f869a784121`.
This directory is vendored in the parent PIE repository.

## Native lifecycle

The public `algorithms/ppo.py` and `runners/on_policy_runner.py` are byte-for-byte
copies of the upstream commit above. Non-PIE tasks use those original classes,
with tensor observations and inference-mode collection.

`PIEOnPolicyRunner` subclasses the native runner interface and owns its
`learn()` and `log()` loops. Collection follows
`act → env.step → process_env_step → compute_returns → update`, using the native
five-element environment return value and `torch.no_grad()` so collected camera
tensors remain usable in CNN backward. Metrics and schema-4 persistence also
belong to the PIE runner; no PIE hooks are installed in the upstream classes.

`PIEPPO` subclasses `PPO`, reusing its initialization, sampling, transition
processing, and returns interface. It owns the full recurrent `update()` loop:
policy/value clipping, entropy, diagonal-Gaussian KL, adaptive learning rate,
Adam, gradient clipping, and joint auxiliary supervision retain their previous
math and order. Actor and supervised gradients both reach the estimator through
one optimizer. The small amount of loop duplication keeps upstream files intact.

`PIERolloutStorage` extends the original transition tensors and inherits its GAE
calculation. An extended `Transition` snapshots sensor inputs before the environment step.
Additional time-by-environment tensors store proprioceptive history, camera
frame indices, supervision, pre-reset successors, and reset masks. Whole
trajectory minibatches carry these tensors directly; no per-step `frames` list
or `actor_batch` wrapper remains. Side-buffer allocations survive rollout clears,
while the frame pool and initial hidden state are reset. The actor has recurrent state; the critic has none.

For timeouts, PIE adds `gamma * V(actual_terminal_critic)` to the reward exactly
once, then native GAE stops its trace at the reset. True failures do not bootstrap.
No reset observation replaces the successor reconstruction target.

## Model and configuration

The task passes native `runner`, `policy`, and `algorithm` dictionaries.
All policy settings are flat fields on `Lite3PIECfgPPO.policy` and explicit
`PIEActorCritic(num_actor_obs, num_critic_obs, num_actions, ...)` arguments,
including encoder/head widths, `init_noise_std`, and actor/critic activation.
Auxiliary weights belong to `Lite3PIECfgPPO.algorithm`. There is no runtime
`ModelConfig` dataclass or duplicate `algorithm.model` alias.

`utils/pie_config.py` converts legacy nested `policy.model_config` only at the
runner/playback boundary. Version-4 saves still write the legacy-compatible
training configuration and effective `model_config`; old and new code can read
the same checkpoints. Missing legacy architecture fields use frozen v4 defaults.
The model's module names, parameter registration order, and construction order
are preserved for model and Adam state compatibility.

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

The task defaults to `camera.input_mode='depth'`: Warp captures real depth,
encodes it, and passes it through the latency queue and two-frame history to
the CNN. Camera pose/FOV randomization is enabled during training and disabled
for playback. Rendering in depth mode does not require the debug flag.

The optional blind-flat ablation uses `camera.input_mode='zero'`: the task directly
supplies a reusable zero image to the history/latency buffers. Training skips
Warp camera initialization, rendering, and encoding. Playback `--show_depth`
sets `camera.render_for_debug=True` to capture real depth for display while
keeping policy input zero. Logical frame IDs, frame pooling, and feature reuse stay active.
CNN parameters remain in Adam. With zero input, the first convolution's weight
gradient is zero, while its bias and deeper network parameters can train.

## Logging, checkpoints, and validation

The PIE runner preserves the original console/TensorBoard reward logging and appends all five auxiliary losses, policy KL, gradient norm, learning rate,
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
The environment snapshot includes `camera.input_mode` and the independent
`domain_rand.randomize_camera` switch. Loading checks the saved input mode before
touching weights or Adam, including inference loads. Existing v4 checkpoints
without the input-mode field are interpreted as `depth`. Playback restores the
saved mode and disables both actuator and camera randomization. The optional
rendering flag is selected by playback CLI rather than inherited from a saved
config; existing v4 zero-mode checkpoints remain valid for resume.

CPU checks pin upstream file hashes and compare two updates against an independent
pre-refactor `3f64242` fixture, including every weight, gradient, Adam moment,
recurrent state, loss, learning rate, and RNG state. See
[`PIE_REFACTOR_REFERENCE.md`](../tests/fixtures/PIE_REFACTOR_REFERENCE.md) for provenance
and reproduction. Other checks exercise native runner ordering, stock PPO behavior,
actor privacy, timeout bootstrap, recurrent resets/replay, joint gradients,
dense/indexed visual equivalence, clipping/scheduling, logs, atomic checkpoints,
and schema-4 resume. GPU simulation, 4096-environment memory use, throughput, and
training convergence require separate GPU validation and are not claimed here.

Install editable into the separate native Conda environment from the parent root:

```bash
python -s -m pip install --no-deps --no-build-isolation -e ./rsl_rl
python -s -m pip install --no-deps --no-build-isolation -e .
```
