# PPO recurrent visual update memory

The 4096-environment formal run exhausted the 4090's 24 GiB VRAM in the first
PPO update. PyTorch reported 19.62 GiB allocated and about 22 MiB reserved but
unallocated, so reserved-memory fragmentation did not explain this failure.

Each of the four trajectory minibatches contains 1024 environments over all
24 rollout steps. The previous update retained the depth CNN's intermediate
activations at every step until the combined recurrent loss was backpropagated.

The depth encoder now uses PyTorch non-reentrant activation checkpointing during
gradient-enabled training. It keeps the inputs and recomputes CNN intermediates
during backward. Collection, recurrent-state refresh, and evaluation still use
the direct encoder. `use_reentrant=False` ensures CNN parameters receive
gradients even though camera inputs do not require gradients.

This changes computation and activation retention, not the network architecture,
weights, observations, reward, PPO loss, or 24-step recurrent gradient boundary.
The formal settings remain 4096 environments, 24 rollout steps, five epochs,
four minibatches, adaptive learning rate, 15000 iterations, and saving every
500 iterations. No mixed precision, shorter sequence, hidden-state detachment,
microbatch split, or physics-buffer change was introduced.

CPU regression checks compare joint PPO/auxiliary updates and parameter
gradients with and without recomputation, including reset masks and input
images that have no gradients. Another check measures retained activation
storage while excluding parameter storage and duplicate references.

Reference: [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html).
