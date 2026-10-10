"""WMP Go1 command groups adapted to PIE's terrain kinds."""
import math

import torch


class PIECommandCurriculum:
    names = ('flat', 'omni', 'stairs', 'forward')
    terrain_groups = {'flat': 0, 'slope': 1, 'stairs': 2, 'gap': 3, 'step': 3, 'hurdle': 3}
    forward_group = 3

    def __init__(self, cfg, group_ids):
        self.group_ids = group_ids
        self.settings = {
            'maxima': {name: list(cfg.group_maxima[name]) for name in self.names},
            'initial_limit': float(cfg.initial_limit),
            'threshold': float(cfg.curriculum_threshold),
            'increment': float(cfg.curriculum_increment),
        }
        self.maxima = torch.tensor([self.settings['maxima'][name] for name in self.names],
                                   device=group_ids.device, dtype=torch.float32)
        if (self.maxima.shape != (len(self.names), 2) or not bool(torch.isfinite(self.maxima).all())
                or bool((self.maxima < 0).any()) or bool((self.maxima[:, 0] <= 0).any())
                or self.maxima[self.forward_group, 1] != 0):
            raise ValueError('Invalid PIE command group maxima')
        initial, threshold, increment = (self.settings[key] for key in
                                         ('initial_limit', 'threshold', 'increment'))
        if (not all(math.isfinite(value) for value in (initial, threshold, increment))
                or initial <= 0 or not 0 < threshold < 1 or increment <= 0):
            raise ValueError('Invalid PIE command curriculum parameters')
        self.initial_limits = self.maxima.clamp_max(initial)
        self.limits = (self.initial_limits if cfg.curriculum else self.maxima).clone()
        self.group_sizes = torch.bincount(group_ids, minlength=len(self.names))
        self.score_sums = torch.zeros(len(self.names), device=group_ids.device)
        self.episode_counts = torch.zeros_like(self.group_sizes)
        self.scores = torch.zeros_like(self.score_sums)

    def update(self, env_ids, episode_lengths, tracking_rewards,
               max_episode_steps, reward_scale):
        """Evaluate finished episodes before the environment clears rewards."""
        if not math.isfinite(reward_scale) or reward_scale <= 0 or max_episode_steps <= 0:
            raise ValueError('Command curriculum requires positive tracking reward normalization')
        active = env_ids[episode_lengths[env_ids] > 0]
        for group in range(len(self.names)):
            ids = active[self.group_ids[active] == group]
            if len(ids) == 0:
                continue
            # WMP uses the full horizon for every episode, including failures.
            self.score_sums[group] += tracking_rewards[ids].sum() / (max_episode_steps * reward_scale)
            self.episode_counts[group] += len(ids)
            if self.episode_counts[group] < self.group_sizes[group]:
                continue
            score = self.score_sums[group] / self.episode_counts[group]
            self.scores[group] = score
            self.score_sums[group] = 0
            self.episode_counts[group] = 0
            if score > self.settings['threshold']:
                self.limits[group] = torch.minimum(
                    self.limits[group] + self.settings['increment'], self.maxima[group])

    def state_dict(self):
        return {'version': 3, 'settings': self.settings,
                **{name: getattr(self, name).detach().cpu().clone() for name in
                   ('limits', 'group_sizes', 'score_sums', 'episode_counts', 'scores')}}

    def load_state_dict(self, state, restore_statistics=True):
        if (not isinstance(state, dict) or state.get('version') != 3
                or state.get('settings') != self.settings):
            raise ValueError('Checkpoint command curriculum settings do not match this task')
        values = {}
        for name in ('limits', 'group_sizes', 'score_sums', 'episode_counts', 'scores'):
            target = getattr(self, name)
            raw = torch.as_tensor(state[name], device=target.device)
            if raw.shape != target.shape or not bool(torch.isfinite(raw).all()):
                raise ValueError('Invalid command curriculum state: ' + name)
            if name in ('group_sizes', 'episode_counts') and (
                    bool((raw < 0).any()) or not torch.equal(raw, raw.long())):
                raise ValueError('Invalid command curriculum counts: ' + name)
            values[name] = raw.to(dtype=target.dtype)
        if bool(((values['limits'] < self.initial_limits)
                 | (values['limits'] > self.maxima)).any()):
            raise ValueError('Checkpoint command limits are outside configured bounds')
        self.limits.copy_(values['limits'])
        self.scores.copy_(values['scores'])
        # Playback may use a different number of environments. Keep learned
        # limits, but discard partial statistics when the population changes.
        self.score_sums.zero_()
        self.episode_counts.zero_()
        if restore_statistics and torch.equal(values['group_sizes'], self.group_sizes):
            self.score_sums.copy_(values['score_sums'])
            self.episode_counts.copy_(values['episode_counts'])
