# PIE refactor reference

`pie_before_rsl_refactor.json.gz` was captured from the unmodified `3f64242`
source, before the native-style refactor. It is gzip-compressed JSON, containing
small CPU tensors, not a policy checkpoint or pickle. It records initialization,
two successive joint updates, gradients, Adam moments/order, GRU states, all PPO
metrics, learning rates, RNG states, and completed iteration counts.

The fixture uses seed 624, the tensor environment in `native_rsl_helpers.py`,
three environments, six rollout steps, two epochs, two minibatches, and the
existing adaptive schedule. The helper and runtime imports must resolve to the
baseline checkout, not the new implementation. To regenerate with the dedicated
native Conda environment:

```bash
mkdir -p /tmp/pie-rsl-reference
# Run from the repository root. This extracts files without switching branches.
git archive 3f64242 rsl_rl legged_gym tests | tar -x -C /tmp/pie-rsl-reference
CUDA_VISIBLE_DEVICES='' python tests/capture_pie_reference.py \
  /tmp/pie-rsl-reference tests/fixtures/pie_before_rsl_refactor.json.gz
```

The exact default-model initialization digest remains covered independently by
`pie_before_cleanup.json`. The new fixture checks all initialized parameters
exactly, and subsequent tensors with FP32 tolerances. The attention key-bias
slice alone uses a 5e-6 absolute parameter tolerance because Adam amplifies its
near-zero floating-point residual; all other parameters use 2e-7 absolute and
1e-5 relative tolerance. RNG states and iteration counters remain exact.

During implementation, independent FP64 baseline/new runs also matched exactly
across all 598 tensors in two updates. The real version-4 `model_500.pt` was
loaded by both implementations: restored Adam states/LR/iteration and three
successive CPU action outputs agreed exactly. A newly saved checkpoint was
also read back by the original code. These are CPU checks and do not establish
GPU simulation, throughput, numerical stability during long training, or
convergence.
