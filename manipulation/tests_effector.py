"""Small dependency-free test for proposal-space conversion (no MuJoCo required)."""
import unittest
from types import SimpleNamespace
import numpy as np
from manipulation.mppi.controller import ControllerConfig, JointMPPIController


class FakeRobot:
    nu = 7
    rollout_arm_actuator_ids = np.arange(7)
    rollout_arm_qpos_adr = np.arange(7)
    model = object()
    mujoco = SimpleNamespace(mj_forward=lambda model, data: None)

    def snapshot(self, data):
        return data

    def new_data(self, snapshot):
        return SimpleNamespace(qpos=np.zeros(7))

    def cartesian_jacobian(self, data):
        return np.array([[1., 0, 0, 0, 0, 0, 0],
                         [0., 1, 0, 0, 0, 0, 0],
                         [0., 0, 1, 0, 0, 0, 0]])


class TestEffectorProjection(unittest.TestCase):
    def test_cartesian_mapping_and_joint_passthrough(self):
        controller = JointMPPIController.__new__(JointMPPIController)
        controller.cfg = ControllerConfig(horizon=4, num_rollouts=2, noise_space='effector',
                                          effector_noise_std=.03)
        controller.robot = FakeRobot()
        controller._joint_std = np.full(7, .2)
        controller._ctrl_low = np.full(7, -2.)
        controller._ctrl_high = np.full(7, 2.)
        nominal = np.zeros((4, 7))
        proposals = np.zeros((2, 4, 7))
        proposals[0, :, 0] = .2
        proposals[1, :, 1] = -.4
        proposals[:, :, 3:] = .8
        mapped = controller._convert_proposals_to_effector_space(proposals, nominal, None)
        np.testing.assert_allclose(mapped[0, :, 0], .03, atol=1e-12)
        np.testing.assert_allclose(mapped[1, :, 1], -.06, atol=1e-12)
        np.testing.assert_allclose(mapped[:, :, 3:], 0., atol=1e-12)
        controller.cfg.noise_space = 'joint'
        self.assertIs(controller._convert_proposals_to_effector_space(proposals, nominal, None), proposals)

    def test_invalid_options(self):
        with self.assertRaises(ValueError):
            ControllerConfig(noise_space='xyz')
        with self.assertRaises(ValueError):
            ControllerConfig(effector_noise_std=0)


if __name__ == '__main__':
    unittest.main()
