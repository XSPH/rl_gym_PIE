"""Inspect the real PIE terrain mesh in the native viewer without a policy."""
from isaacgym import gymapi, gymutil

import numpy as np

from legged_gym.utils.terrain import PIETerrain


def preview_terrain(cfg, args):
    if args.headless or args.graphics_device_id < 0:
        raise ValueError("Terrain preview requires a viewer and graphics device")
    if args.checkpoint_file is not None or args.show_depth:
        raise ValueError("Terrain preview uses the current terrain config; omit --checkpoint_file and --show_depth")
    if not 0 <= args.terrain_level < cfg.terrain.num_rows:
        raise ValueError("--terrain_level must select an existing terrain row")
    if args.terrain_column is not None and not 0 <= args.terrain_column < cfg.terrain.num_cols:
        raise ValueError("--terrain_column must select an existing terrain column")
    seed = args.seed if args.seed is not None else getattr(cfg, 'seed', 1)
    print("[PIE terrain preview] Building current {}x{} map, seed={} ...".format(
        cfg.terrain.num_rows, cfg.terrain.num_cols, seed), flush=True)
    terrain = PIETerrain(cfg.terrain, 0, seed=seed)
    print("[PIE terrain preview] vertices={}, triangles={}, geometry_version={}".format(
        len(terrain.vertices), len(terrain.triangles), cfg.terrain.geometry_version), flush=True)
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = cfg.sim.dt
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    # Only a static mesh is rendered. No robot, Warp camera, policy or rollout
    # is allocated, and no physics stepping is required.
    params.use_gpu_pipeline = False
    params.physx.use_gpu = False
    params.physx.num_threads = args.num_threads
    sim, viewer = None, None
    try:
        sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, params)
        if sim is None:
            raise RuntimeError("Failed to create Isaac Gym terrain preview simulation")
        mesh = gymapi.TriangleMeshParams()
        mesh.nb_vertices = len(terrain.vertices)
        mesh.nb_triangles = len(terrain.triangles)
        mesh.static_friction = cfg.terrain.static_friction
        mesh.dynamic_friction = cfg.terrain.dynamic_friction
        mesh.restitution = cfg.terrain.restitution
        gym.add_triangle_mesh(sim, terrain.vertices.reshape(-1), terrain.triangles.reshape(-1), mesh)
        marker_env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
        gym.prepare_sim(sim)
        camera = gymapi.CameraProperties()
        camera.width, camera.height = 1280, 800
        viewer = gym.create_viewer(sim, camera)
        if viewer is None:
            raise RuntimeError("Failed to open Isaac Gym viewer; check DISPLAY and NVIDIA graphics")
        keys = ((gymapi.KEY_ESCAPE, 'quit'), (gymapi.KEY_LEFT, 'previous_column'),
                (gymapi.KEY_RIGHT, 'next_column'), (gymapi.KEY_UP, 'higher_level'),
                (gymapi.KEY_DOWN, 'lower_level'), (gymapi.KEY_O, 'overview'),
                (gymapi.KEY_R, 'selected'))
        for key, action in keys:
            gym.subscribe_viewer_keyboard_event(viewer, key, action)
        for number in range(1, len(cfg.terrain.kinds) + 1):
            gym.subscribe_viewer_keyboard_event(viewer, getattr(gymapi, 'KEY_' + str(number)),
                                               'kind_' + str(number - 1))
        row = args.terrain_level
        column = (args.terrain_column if args.terrain_column is not None else
                  (terrain.kinds.index('gap') if 'gap' in terrain.kinds else 0))
        print("[PIE terrain preview] Left/Right: column; Up/Down: level; "
              "1-6: terrain type; O: whole map; R: selected tile; Esc: close", flush=True)
        print("[PIE terrain preview] Types: " + ', '.join(
            '{}={}'.format(i + 1, kind) for i, kind in enumerate(cfg.terrain.kinds)), flush=True)

        def focus(overview=False):
            origin = terrain.env_origins[row, column]
            if overview:
                length = cfg.terrain.num_rows * cfg.terrain.terrain_length
                width = cfg.terrain.num_cols * cfg.terrain.terrain_width
                target = gymapi.Vec3(length / 2, width / 2, 0)
                eye = gymapi.Vec3(-length / 2, -width / 3, max(length, width))
            else:
                target = gymapi.Vec3((row + 0.5) * cfg.terrain.terrain_length,
                                    (column + 0.5) * cfg.terrain.terrain_width, float(origin[2]))
                eye = gymapi.Vec3(row * cfg.terrain.terrain_length - 3,
                                 column * cfg.terrain.terrain_width - 4, max(5, float(origin[2]) + 6))
            gym.viewer_camera_look_at(viewer, None, eye, target)
            gym.clear_lines(viewer)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(float(origin[0]), float(origin[1]), float(origin[2]) + cfg.init_state.pos[2])
            sphere = gymutil.WireframeSphereGeometry(0.15, 12, 12, None, color=(1, 0.8, 0.1))
            gymutil.draw_lines(sphere, gym, viewer, marker_env, pose)
            # A one-metre +X arrow marks the nominal spawn and travel direction.
            start = np.asarray([origin[0], origin[1], origin[2] + 0.12], dtype=np.float32)
            end = start + [1, 0, 0]
            lines = np.asarray([start, end, end, end + [-.2, .15, 0],
                                end, end + [-.2, -.15, 0]], dtype=np.float32)
            colors = np.tile(np.asarray([1, .8, .1], dtype=np.float32), (3, 1))
            gym.add_lines(viewer, marker_env, 3, lines, colors)
            print("[PIE terrain preview] level={} column={} kind={} spawn=({:.2f}, {:.2f}, {:.2f}){}".format(
                row, column, terrain.kinds[column], *origin,
                ' overview' if overview else ''), flush=True)

        focus()
        quit_requested = False
        while not quit_requested and not gym.query_viewer_has_closed(viewer):
            for event in gym.query_viewer_action_events(viewer):
                if event.value <= 0:
                    continue
                action = event.action
                if action == 'quit':
                    quit_requested = True
                    break
                if action == 'previous_column':
                    column = (column - 1) % cfg.terrain.num_cols
                elif action == 'next_column':
                    column = (column + 1) % cfg.terrain.num_cols
                elif action == 'higher_level':
                    row = min(row + 1, cfg.terrain.num_rows - 1)
                elif action == 'lower_level':
                    row = max(row - 1, 0)
                elif action.startswith('kind_'):
                    kind = cfg.terrain.kinds[int(action.split('_')[1])]
                    if kind not in terrain.kinds:
                        print('[PIE terrain preview] No column for ' + kind, flush=True)
                        continue
                    column = terrain.kinds.index(kind)
                focus(overview=action == 'overview')
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
        print('[PIE terrain preview] Viewer closed', flush=True)
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        if sim is not None:
            gym.destroy_sim(sim)
