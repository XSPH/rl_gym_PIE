"""Optional playback window for captured depth and the policy's frame history."""
import time

import numpy as np
import torch
from rsl_rl.utils.pie_config import depth_input_mode


class DepthViewer:
    """Read existing camera buffers without capturing or advancing sensors."""

    def __init__(self, camera_cfg, num_envs, env_id=0, control_dt=0.02):
        if not 0 <= env_id < num_envs:
            raise ValueError("--depth_env must be between 0 and num_envs - 1")
        if camera_cfg.history != 2:
            raise ValueError("The PIE depth window expects two policy history frames")
        input_mode = depth_input_mode(camera_cfg)
        try:
            import tkinter as tk
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError as error:
            raise RuntimeError("--show_depth requires matplotlib and tkinter") from error

        self._tk = tk
        try:
            self._window = tk.Tk()
        except tk.TclError as error:
            raise RuntimeError(
                "--show_depth needs a graphical desktop/display. "
                "--headless only disables the Isaac Gym viewer."
            ) from error
        self._closed = False
        self._env_id, self._num_envs = env_id, num_envs
        self._near, self._far = camera_cfg.near, camera_cfg.far
        self._normalized = camera_cfg.normalize
        self._input_mode = input_mode
        self._period = max(control_dt * camera_cfg.update_every, 1.0 / 30.0)
        self._next_draw = 0.0
        self._last_capture = None
        self._dirty = True
        self._window.title("PIE depth camera")
        self._window.protocol("WM_DELETE_WINDOW", self.close)
        self._window.bind("<KeyPress>", self._on_key)
        try:
            self._figure = Figure(figsize=(12, 4.3), dpi=100)
            self._figure.subplots_adjust(left=0.025, right=0.92, top=0.75,
                                         bottom=0.25, wspace=0.22)
            axes = self._figure.subplots(1, 3)
            units = "normalized input" if self._normalized else "depth (m)"
            if input_mode == 'zero':
                units = "normalized input; zero mode" if self._normalized else "zero input; no distance"
            labels = ("Fresh capture (m)", "Policy history: older\n" + units,
                      "Policy history: newest\n" + units)
            self._images = []
            for index, (axis, label) in enumerate(zip(axes, labels)):
                dimensionless = index > 0 and (self._normalized or input_mode == 'zero')
                limits = (-0.5, 0.5) if dimensionless else (self._near, self._far)
                blank = np.full((camera_cfg.height, camera_cfg.width),
                                0.0 if dimensionless else self._far)
                self._images.append(axis.imshow(blank, cmap="turbo_r",
                                               vmin=limits[0], vmax=limits[1],
                                               interpolation="nearest"))
                axis.set_title(label, fontsize=11)
                axis.set_axis_off()
            self._figure.colorbar(self._images[0], ax=axes[0], fraction=0.046,
                                 pad=0.04).set_label("Depth (m)")
            self._figure.colorbar(self._images[1], ax=list(axes[1:]), fraction=0.025,
                                 pad=0.04).set_label(units)
            self._title = self._figure.suptitle("Waiting for camera frames", fontsize=12)
            self._figure.text(0.5, 0.09,
                              "Left / Right or P / N: switch robot    Esc / Q: close depth window\n"
                              "Policy images show network values with configured delay; zero mode carries no distance.",
                              ha="center", fontsize=9)
            self._canvas = FigureCanvasTkAgg(self._figure, master=self._window)
            widget = self._canvas.get_tk_widget()
            widget.pack(fill=tk.BOTH, expand=True)
            widget.focus_set()
            self._canvas.draw()
        except Exception:
            self.close()
            raise

    def _on_key(self, event):
        key = event.keysym.lower()
        if key in ("escape", "q"):
            self.close()
        elif key in ("right", "n", "left", "p"):
            direction = 1 if key in ("right", "n") else -1
            self._env_id = (self._env_id + direction) % self._num_envs
            self._dirty = True

    def update(self, env):
        """Call before policy inference to show precisely its current history."""
        if self._closed:
            return
        try:
            self._window.update_idletasks()
            self._window.update()
        except self._tk.TclError:
            self._closed = True
        if self._closed:
            return
        now = time.monotonic()
        capture = env.camera_capture_serial
        if not self._dirty and (capture == self._last_capture or now < self._next_draw):
            return
        # Transfer only the selected robot's three existing frames. Never call
        # camera.render(), encode(), or any sensor history update here.
        frames = torch.cat((env.camera.depth[self._env_id, None],
                            env.depth_history[self._env_id]), dim=0).detach().cpu().numpy()
        for artist, frame in zip(self._images, frames):
            artist.set_data(frame)
        column = int(env.terrain_types[self._env_id])
        kind = env.atlas.kinds[column]
        self._title.set_text("Robot {} ({} total)  |  {}  |  {} x {} pixels".format(
            self._env_id, self._num_envs, kind, frames.shape[2], frames.shape[1]))
        self._window.title("PIE depth camera - robot {} ({})".format(self._env_id, kind))
        self._canvas.draw()
        self._last_capture = capture
        self._next_draw = time.monotonic() + self._period
        self._dirty = False

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._window.destroy()
        except self._tk.TclError:
            pass
