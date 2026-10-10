"""CPU regression checks for the WMP command curriculum and current state."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from native_cpu_helpers import load_native_classes


def curriculum(groups):
    cfg = SimpleNamespace(curriculum=True, initial_limit=.5,
                          curriculum_threshold=.8, curriculum_increment=.2,
                          group_maxima={'flat': [2., 1.], 'omni': [1.5, 1.],
                                        'stairs': [1., .8], 'forward': [1., 0.]})
    command_class = load_native_classes().task._init_command_curriculum.__globals__['PIECommandCurriculum']
    return command_class(cfg, torch.tensor(groups))


def test_command_groups_advance_independently_and_stop_at_their_caps():
    course = curriculum([0, 0, 1, 2, 3, 3])
    lengths = torch.full((6,), 1000)
    rewards = torch.full((6,), 27.)
    course.update(torch.tensor([0]), lengths, rewards, 1000, .03)
    assert course.episode_counts[0] == 1 and course.limits[0, 0] == .5
    course.update(torch.tensor([1]), lengths, rewards, 1000, .03)
    torch.testing.assert_close(course.limits[0], torch.tensor([.7, .7]))
    torch.testing.assert_close(course.limits[1:], torch.tensor([[.5, .5], [.5, .5], [.5, 0.]]))
    for _ in range(10):
        course.update(torch.arange(6), lengths, rewards, 1000, .03)
    torch.testing.assert_close(course.limits, course.maxima)


def test_early_failures_use_full_horizon_and_zero_step_resets_do_not_count():
    course = curriculum([3, 3])
    lengths = torch.tensor([0, 100])
    rewards = torch.tensor([30., 3.])
    course.update(torch.arange(2), lengths, rewards, 1000, .03)
    assert course.episode_counts[3] == 1
    lengths[0] = 1000
    course.update(torch.tensor([0]), lengths, rewards, 1000, .03)
    torch.testing.assert_close(course.scores[3], torch.tensor(.55))
    torch.testing.assert_close(course.limits[3], torch.tensor([.5, 0.]))


def test_curriculum_state_restores_limits_and_drops_stats_when_population_changes():
    original = curriculum([0, 0, 3, 3])
    original.limits[0] = torch.tensor([.7, .7])
    original.update(torch.tensor([0]), torch.full((4,), 1000), torch.full((4,), 27.), 1000, .03)
    saved = original.state_dict()
    matching = curriculum([0, 0, 3, 3])
    matching.load_state_dict(saved)
    torch.testing.assert_close(matching.limits, original.limits)
    torch.testing.assert_close(matching.score_sums, original.score_sums)
    different = curriculum([0, 0, 0, 3, 3])
    different.load_state_dict(saved)
    torch.testing.assert_close(different.limits, original.limits)
    assert not different.score_sums.any() and not different.episode_counts.any()
    old = deepcopy(saved)
    old['version'] = 2
    with pytest.raises(ValueError, match='settings'):
        matching.load_state_dict(old)
    with pytest.raises(ValueError, match='settings'):
        matching.load_state_dict(None)
