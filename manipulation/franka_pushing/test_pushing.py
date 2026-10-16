"""Pure-Python invariants for the pushing task (no MuJoCo required)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np

from manipulation.franka_pushing.run import build_controller, make_parser
from manipulation.mppi.rollout import RolloutCostConfig, rollout_controls
from manipulation.tracks import StraightTrack


class _Data:
    def __init__(self):
        self.time = 0.0
        self.ctrl = np.zeros(2, dtype=float)
        self.box_x = 0.0


class _BoxRobot:
    task_body_id = 2
    root_body_id = 1
    task_rest_height = 0.0
    task_half_height = 0.05
    physics_dt = 0.01
    fused_task_mode = 4
    nu = 2
    model = object()

    def __init__(self):
        self.mujoco = SimpleNamespace(mj_step=lambda _model, data: self._step(data))

    @staticmethod
    def _step(data):
        data.time += 0.01
        data.box_x += 0.10

    def new_data(self, _snapshot):
        return _Data()

    @staticmethod
    def control_scale():
        return np.array([2.0, 4.0])

    @staticmethod
    def rate_limited_ctrl(u, *, control_dt, data):
        return np.asarray(u)

    @staticmethod
    def task_xy(data):
        return np.array([data.box_x, 0.0])

    @staticmethod
    def xy(data):
        return np.array([0.0, 0.0])

    @staticmethod
    def task_xyz(data):
        return np.array([data.box_x, 0.0, 0.0])

    @staticmethod
    def goal_xyz(data):
        return np.array([0.4, 0.0, 0.0])

    @staticmethod
    def tip_box_surface_distance(data):
        return 0.05

    @staticmethod
    def task_height(data):
        return 100.0  # Intentionally violates lift thresholds.

    @staticmethod
    def task_up(data):
        return -1.0  # Intentionally violates tip thresholds.

    @staticmethod
    def root_up(data):
        return -1.0  # Intentionally violates orientation weight.

    @staticmethod
    def has_fallen(data, *, flipped_threshold=0):
        return False

    @staticmethod
    def has_table_contact(data):
        return True  # Ignored in mode-4 cost.


class PushingUnitTests(TestCase):
    def test_only_three_terms_python_cost(self):
        robot = _BoxRobot()
        track = StraightTrack(origin_xy=(0, 0), direction=(1, 0), road_width=0.1)
        # Deliberately turn on unrelated terms to prove mode 4 ignores them.
        cfg = RolloutCostConfig(
            robot_box_approach_weight=2.0, target_error_weight=4.0,
            control_deviation_weight=0.1, upright_weight=999.0,
            table_contact_weight=999.0, non_tip_contact_weight=999.0,
            box_progress_weight=999.0, box_lift_weight=999.0,
            box_max_lift=0, box_min_up=1.0,
        )
        controls = np.array([[1.0, 2.0], [1.0, 2.0]])
        positions, cost, _, failed = rollout_controls(
            robot, None, controls, track, current_s=0.0,
            control_substeps=1, nominal_controls=np.zeros_like(controls), cost_cfg=cfg,
        )
        # box distances: .30 and .20; tip distance always .05;
        # normalized per-actuator squared control deviation is .25.
        expected = 2.0 * 0.05 + 4.0 * 0.25 + 0.1 * 0.25
        self.assertFalse(failed)
        np.testing.assert_allclose(positions, [[0.1, 0], [0.2, 0]], atol=1e-12)
        self.assertAlmostEqual(cost, expected, places=12)

    def test_task_configuration_fixed_noise_and_no_other_weights(self):
        with patch("manipulation.franka_pushing.run.JointMPPIController") as maker:
            build_controller(object(), None)
            cfg = maker.call_args.args[3]
        self.assertFalse(cfg.adaptive_noise)
        self.assertEqual(cfg.noise_min_fraction, 1.0)
        self.assertEqual(cfg.robot_box_approach_weight, 1.0)
        self.assertEqual(cfg.target_error_weight, 1.0)
        self.assertEqual(cfg.control_deviation_weight, 1e-4)
        for field in (
            "upright_weight", "box_progress_weight", "robot_progress_weight",
            "box_lift_weight", "terminal_target_weight", "overshoot_weight",
            "terminal_overshoot_weight", "target_velocity_weight",
            "non_tip_contact_weight", "tip_contact_reward", "tip_push_reward",
            "table_contact_weight", "box_speed_weight", "terminal_velocity_weight",
            "command_derivative_weight", "reach_terminal_weight",
            "reach_velocity_weight", "reach_region_weight", "reach_settle_weight",
            "reach_acceleration_weight",
        ):
            self.assertEqual(getattr(cfg, field), 0.0, field)

    def test_command_line_defaults(self):
        args = make_parser().parse_args(["--no-viewer"])
        self.assertEqual(tuple(args.start_xy), (0.55, 0.0))
        self.assertEqual(tuple(args.goal_xy), (0.55, 0.18))
        self.assertEqual(args.sampling, "spike")
        self.assertEqual(args.noise_space, "effector")



class WaypointUnitTests(TestCase):
    def test_default_square_and_cli(self):
        from manipulation.franka_pushing.waypoints import square_corners, resolve_waypoints
        expected = np.array([
            [0.55, 0.00], [0.55, 0.18], [0.73, 0.18], [0.73, 0.00]
        ])
        np.testing.assert_allclose(square_corners((0.55, 0.0), (0.55, 0.18)), expected)
        args = make_parser().parse_args(["--no-viewer"])
        self.assertEqual(args.route, "square")
        self.assertEqual(args.cycles, 2)
        self.assertIsNone(args.waypoints)
        np.testing.assert_allclose(
            resolve_waypoints(start_xy=args.start_xy, goal_xy=args.goal_xy,
                              route=args.route, waypoints=args.waypoints), expected
        )
        custom = np.array([[.4, -.1], [.4, .1], [.6, .1], [.6, -.1]])
        np.testing.assert_allclose(
            resolve_waypoints(start_xy=(0, 0), goal_xy=(0, 1),
                              route="pingpong", waypoints=custom.reshape(-1)), custom
        )
        with self.assertRaises(ValueError):
            resolve_waypoints(start_xy=(0, 0), goal_xy=(0, 0), route="square")
        with self.assertRaises(ValueError):
            resolve_waypoints(start_xy=(0, 0), goal_xy=(0, 1),
                              route="point", waypoints=custom.reshape(-1))

    def test_square_two_cycles(self):
        from manipulation.franka_pushing.waypoints import WaypointRoute
        route = WaypointRoute("square", cycles=2)
        arrived = []
        for i in range(8):
            self.assertEqual(route.goal_index, (1, 2, 3, 0)[i % 4])
            arrived.append(route.observe(distance=0.001, dt=0.2,
                                        tolerance=0.025, hold_seconds=0.2))
        self.assertEqual(arrived, [1, 2, 3, 0, 1, 2, 3, 0])
        self.assertEqual(route.cycles_completed, 2)
        self.assertEqual(route.segments_completed, 8)
        self.assertTrue(route.done)
        self.assertIsNone(route.observe(distance=0.001, dt=0.2,
                                        tolerance=0.025, hold_seconds=0.2))

    def test_pingpong_continuous_and_dwell_reset(self):
        from manipulation.franka_pushing.waypoints import WaypointRoute
        route = WaypointRoute("pingpong", cycles=0)
        self.assertIsNone(route.observe(distance=0.01, dt=0.1, tolerance=0.025,
                                        hold_seconds=0.3))
        self.assertIsNone(route.observe(distance=0.2, dt=0.1, tolerance=0.025,
                                        hold_seconds=0.3))
        self.assertEqual(route.dwell_seconds, 0)
        self.assertIsNone(route.observe(distance=0.01, dt=0.2, tolerance=0.025,
                                        hold_seconds=0.3))
        arrived = [route.observe(distance=0.01, dt=0.1, tolerance=0.025,
                                 hold_seconds=0.3)]
        for i in range(11):
            arrived.append(route.observe(distance=0.0, dt=0.3, tolerance=0.025,
                                         hold_seconds=0.3))
        self.assertEqual(arrived, [1, 2, 3, 2, 1, 0] * 2)
        self.assertEqual(route.cycles_completed, 2)
        self.assertFalse(route.done)

    def test_original_single_goal_can_be_selected(self):
        from manipulation.franka_pushing.waypoints import WaypointRoute, resolve_waypoints
        p = resolve_waypoints(start_xy=(0, 0), goal_xy=(0, 1), route="point")
        self.assertEqual(p.shape, (2, 2))
        route = WaypointRoute("point", cycles=0)
        self.assertEqual(route.goal_index, 1)
        self.assertEqual(route.observe(distance=0., dt=1., tolerance=.01, hold_seconds=.2), 1)
        self.assertTrue(route.done)

    def test_retarget_syncs_native_rollout_and_resets_plan(self):
        from manipulation.franka_pushing.run import _retarget_goal
        from unittest.mock import Mock
        robot = Mock()
        a, b, c = object(), object(), object()
        batcher = SimpleNamespace(data=[a, b], _state_scratch=c)
        controller = SimpleNamespace(
            rollout_batcher=batcher, nominal_batcher=batcher,
            sync_planning_model=Mock(), reset_warm_start=Mock(),
        )
        _retarget_goal(robot, controller, [.73, .18])
        robot.set_goal_xy.assert_called_once_with([.73, .18])
        self.assertEqual(robot.sync_goal_mocap.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in robot.sync_goal_mocap.call_args_list],
            [a, b, c],
        )
        controller.sync_planning_model.assert_called_once_with()
        controller.reset_warm_start.assert_called_once_with()



class FullRouteIntegrationUnitTests(TestCase):
    """Exercise switching, NPZ traces, and rewind without MuJoCo hardware."""

    def test_headless_mocked_square_records_all_goal_changes(self):
        import importlib
        import tempfile
        from pathlib import Path

        run_module = importlib.import_module("manipulation.franka_pushing.run")

        class FakeRobot:
            name = "franka-push"
            physics_dt = 0.01
            nu = 2
            _joint_qadr = np.array([0])
            _joint_dadr = np.array([0])

            def __init__(self, *, start_xy, goal_xy, **_kwargs):
                self.box_xy = np.asarray(start_xy, dtype=float).copy()
                self.goal_xy = np.asarray(goal_xy, dtype=float).copy()
                self.start_xy = self.box_xy.copy()
                self.data = self
                self.model = object()
                self.time = 0.0
                self.ctrl = np.zeros(2)

            def set_waypoints_xy(self, xy):
                self.waypoints = np.asarray(xy).copy()

            def hide_waypoints(self):
                pass

            def goal_xyz(self):
                return np.array([*self.goal_xy, 0.385])

            def start_xyz(self):
                return np.array([*self.start_xy, 0.385])

            def task_xyz(self):
                return np.array([*self.box_xy, 0.385])

            def task_velocity(self):
                return np.zeros(3)

            def tip_xyz(self):
                return np.array([.55, .0, .4])

            def tip_box_surface_distance(self):
                return 0.05

            def box_goal_distance(self):
                return float(np.linalg.norm(self.box_xy - self.goal_xy))

            def set_goal_xy(self, xy):
                self.goal_xy = np.asarray(xy, dtype=float).copy()

            def snapshot(self):
                return SimpleNamespace(
                    time=self.time, qpos=np.array([self.box_xy[0]]),
                    qvel=np.array([0.0]), act=np.zeros(0), ctrl=self.ctrl.copy(),
                )

            def step_control(self, action, *, substeps):
                self.ctrl[:] = action
                self.box_xy = self.goal_xy.copy()  # Impossibly perfect physics; route only.
                self.time += .01

        class FakeController:
            rollout_backend_name = "cpu/fused/mock"
            control_substeps = 1
            rollout_batcher = None
            nominal_batcher = None

            def __init__(self):
                self.syncs = 0
                self.resets = 0

            def step(self, _data, _progress):
                return np.ones(2), {"timing_ms": {"total": 0.1}}

            def sync_planning_model(self):
                self.syncs += 1

            def reset_warm_start(self):
                self.resets += 1

            def close(self):
                pass

        ctrl = FakeController()
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "route.npz"
            args = make_parser().parse_args([
                "--cycles", "1", "--steps", "20", "--hold-seconds", ".01",
                "--no-viewer", "--analysis-trace", "--save", str(file),
            ])
            with patch.object(run_module, "FrankaPushRobot", FakeRobot), \
                 patch.object(run_module, "build_controller", return_value=ctrl):
                result = run_module.run(args)
            self.assertTrue(result["completed"])
            self.assertEqual(result["waypoints_reached"], 4)
            self.assertEqual(result["cycles_completed"], 1)
            self.assertEqual(ctrl.syncs, 3)  # No retarget after the final arrival.
            self.assertEqual(ctrl.resets, 3)
            with np.load(file, allow_pickle=False) as z:
                self.assertEqual(z["replay_format_version"], 5)
                np.testing.assert_array_equal(z["arrival_waypoint_indices"], [1, 2, 3, 0])
                np.testing.assert_array_equal(z["arrival_step_indices"], [1, 2, 3, 4])
                np.testing.assert_array_equal(z["goal_waypoint_index_history"], [1, 2, 3, 0, 0])
                np.testing.assert_allclose(z["goal_xyz_history"][:, :2], [
                    [.55, .18], [.73, .18], [.73, 0], [.55, 0], [.55, 0],
                ])
                self.assertEqual(z["controls"].shape, (4, 2))
                self.assertEqual(z["box_xyz"].shape, (5, 3))
                self.assertEqual(z["goal_xyz_history"].shape[0], z["qpos"].shape[0])
                np.testing.assert_array_equal(z["noise_scale"], np.ones(4))

    def test_replay_applies_saved_goal_for_each_frame(self):
        from manipulation.experiments import replay

        class FakeRobot:
            def __init__(self):
                self.model = SimpleNamespace(na=0)
                self.data = SimpleNamespace(
                    time=0., qpos=np.zeros(1), qvel=np.zeros(1),
                    ctrl=np.zeros(1),
                )
                self.nu = 1
                self.default_ctrl = np.zeros(1)
                self.xy = np.array([0., 0.])
                self.mujoco = SimpleNamespace(mj_forward=lambda _m, _d: None)
                self.goals = []

            def goal_xyz(self):
                return np.array([*self.xy, .385])

            def set_goal_xy(self, xy):
                self.xy = np.asarray(xy, dtype=float).copy()
                self.goals.append(self.xy.copy())

            def clip_ctrl(self, u):
                return np.asarray(u)

        robot = FakeRobot()
        history = np.array([[.55, .18, .385], [.73, .18, .385], [.55, 0, .385]])
        with patch.object(replay, "FrankaPushRobot", FakeRobot):
            for i in range(len(history)):
                replay._set_recorded_state(
                    robot, qpos=np.zeros((3, 1)), qvel=np.zeros((3, 1)),
                    state_time=np.arange(3.), act=np.zeros((3, 0)),
                    controls=np.zeros((2, 1)), i=i, goal_history=history,
                )
        self.assertEqual(len(robot.goals), 3)
        np.testing.assert_allclose(robot.xy, [0.55, 0.0])


if __name__ == "__main__":
    main()
