"""Inspect real Matplotlib artists with an in-memory window, without a GUI."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from native_cpu_helpers import load_native_classes, module


@pytest.mark.parametrize('input_mode,normalized', [('zero', True), ('zero', False),
                                                ('depth', True), ('depth', False)])
def test_depth_window_keeps_raw_metres_and_exact_policy_input_values(monkeypatch, input_mode, normalized):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    class Window:
        def title(self, value):
            self.caption = value
        def protocol(self, *args): pass
        def bind(self, *args): pass
        def update_idletasks(self): pass
        def update(self): pass
        def destroy(self): pass
    class Canvas:
        def __init__(self, figure, master):
            self.canvas = FigureCanvasAgg(figure)
        def get_tk_widget(self):
            return NS(pack=lambda **kwargs: None, focus_set=lambda: None)
        def draw(self):
            self.canvas.draw()
    monkeypatch.setitem(sys.modules, 'tkinter', module('tkinter', Tk=Window,
                         TclError=RuntimeError, BOTH='both'))
    monkeypatch.setitem(sys.modules, 'matplotlib.backends.backend_tkagg',
                         module('matplotlib.backends.backend_tkagg', FigureCanvasTkAgg=Canvas))
    path = Path(__file__).parents[1] / 'legged_gym/utils/depth_viewer.py'
    spec = importlib.util.spec_from_file_location('pie_cpu_depth_viewer', path)
    viewer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewer_module)
    cfg = load_native_classes().config().camera
    cfg.input_mode, cfg.normalize, cfg.height, cfg.width = input_mode, normalized, 8, 8
    viewer = viewer_module.DepthViewer(cfg, 1)
    raw = torch.linspace(cfg.near, cfg.far, 64).reshape(1, 8, 8)
    policy = torch.zeros(1, 2, 8, 8)
    if input_mode == 'depth':
        policy[:] = ((raw[:, None] - cfg.near) / (cfg.far - cfg.near) - 0.5
                     if normalized else raw[:, None])
    env = NS(camera=NS(depth=raw), depth_history=policy, camera_capture_serial=1,
             terrain_types=torch.zeros(1, dtype=torch.long), atlas=NS(kinds=['flat']))
    try:
        viewer.update(env)
        np.testing.assert_array_equal(viewer._images[0].get_array(), raw[0].numpy())
        for index in (1, 2):
            np.testing.assert_array_equal(viewer._images[index].get_array(), policy[0, index-1].numpy())
        titles = [axis.get_title() for axis in viewer._figure.axes[:3]]
        assert 'Fresh capture (m)' == titles[0]
        if normalized:
            assert all('normalized input' in title for title in titles[1:])
        if input_mode == 'zero':
            assert all('zero' in title for title in titles[1:])
            assert viewer._images[1].get_clim() == (-0.5, 0.5)
        assert viewer._figure.axes[3].get_ylabel() == 'Depth (m)'
        assert env.camera_capture_serial == 1
        assert torch.equal(env.camera.depth, raw)
        assert torch.equal(env.depth_history, policy)
        viewer.update(env)
        assert viewer._last_capture == 1
    finally:
        viewer.close()
