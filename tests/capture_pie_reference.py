"""Capture the independent baseline; run against the unmodified 3f64242 checkout.

Usage: python tests/capture_pie_reference.py BASELINE_ROOT OUTPUT.json.gz
This is a fixture generator, not part of ordinary pytest collection.
"""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline_root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.baseline_root.resolve()
    sys.path[:0] = [str(root / 'rsl_rl'), str(root / 'tests'), str(root)]

    import torch
    from native_rsl_helpers import TensorEnvironment, train_config, algorithm_config
    from rsl_rl.runners import PIEOnPolicyRunner
    from rsl_rl.modules import actor_critic_pie

    if not hasattr(actor_critic_pie, 'ModelConfig'):
        raise RuntimeError('Capture requires the original 3f64242 implementation, not the refactor')

    torch.set_num_threads(1)
    torch.manual_seed(624)
    runner = PIEOnPolicyRunner(TensorEnvironment(), train_config(
        cfg=algorithm_config(num_learning_epochs=2, num_mini_batches=2), rollout=6), device='cpu')
    reference = {'initial': deepcopy(runner.alg.actor_critic.state_dict()), 'updates': []}
    for _ in range(2):
        runner.learn(1)
        model = runner.alg.actor_critic
        reference['updates'].append(deepcopy({
            'model': model.state_dict(),
            'gradients': {name: param.grad for name, param in model.named_parameters() if param.grad is not None},
            'optimizer': runner.alg.optimizer.state_dict(), 'hidden': model.get_hidden_states()[0],
            'metrics': runner.alg.metrics, 'learning_rate': runner.alg.learning_rate,
            'rng': torch.get_rng_state(), 'iteration': runner.current_learning_iteration,
        }))

    def encode(value):
        if isinstance(value, torch.Tensor):
            return {'dtype': str(value.dtype).split('.')[1], 'tensor': value.tolist()}
        if isinstance(value, dict):
            return {str(name): encode(item) for name, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        return value

    payload = {'source_commit': '3f64242', 'seed': 624, 'torch_version': str(torch.__version__),
               'reference': encode(reference)}
    args.output.write_bytes(gzip.compress(json.dumps(payload, separators=(',', ':')).encode(), mtime=0))


if __name__ == '__main__':
    main()
