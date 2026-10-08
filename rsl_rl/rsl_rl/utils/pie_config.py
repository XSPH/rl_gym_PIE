"""PIE camera input semantics shared by task and checkpoint consumers."""


def depth_input_mode(camera_cfg):
    """Missing fields in existing version-4 checkpoints mean real depth."""
    if isinstance(camera_cfg, dict):
        mode = camera_cfg.get('input_mode', 'depth')
    else:
        mode = getattr(camera_cfg, 'input_mode', 'depth')
    if mode not in ('depth', 'zero'):
        raise ValueError("camera.input_mode must be 'depth' or 'zero'; got {!r}".format(mode))
    return mode
