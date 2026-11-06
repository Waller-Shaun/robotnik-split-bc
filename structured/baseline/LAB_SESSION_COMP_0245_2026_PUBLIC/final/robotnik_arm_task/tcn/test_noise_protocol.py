"""Noise-profile, native sampling, and ground-truth isolation regression tests."""
import copy
import json
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np
from rollout_noisy import NOISY_CONFIG, NOISE_KEYS, ROOT, ground_truth_ee, validate_noise_config


class NoiseProtocolTests(unittest.TestCase):
    def test_screenshot_config_exactly_matches_eight_requested_scales(self):
        config = validate_noise_config(NOISY_CONFIG)
        self.assertEqual(config['robot_pybullet']['noise_flag'], [1])
        self.assertEqual(config['robot_pybullet']['robot_noise'],
                         [{key: .001 for key in NOISE_KEYS}])

    def test_only_two_noise_fields_differ_from_historical_configuration(self):
        original = json.loads((ROOT/'artifacts/RoboEnv/configs/robotnik_arm.json').read_text())
        noisy = json.loads(NOISY_CONFIG.read_text())
        self.assertEqual(original['robot_pybullet']['noise_flag'], [0])
        for key in ('noise_flag', 'robot_noise'):
            noisy['robot_pybullet'][key] = original['robot_pybullet'][key]
        self.assertEqual(noisy, original)

    def test_malformed_or_disabled_profiles_fail_loudly(self):
        config = json.loads(NOISY_CONFIG.read_text())
        variants = []
        disabled = copy.deepcopy(config)
        disabled['robot_pybullet']['noise_flag'] = [0]
        variants.append(disabled)
        wrong_scale = copy.deepcopy(config)
        wrong_scale['robot_pybullet']['robot_noise'][0]['joint_cov'] = .0001
        variants.append(wrong_scale)
        missing = copy.deepcopy(config)
        del missing['robot_pybullet']['robot_noise'][0]['base_ori_cov']
        variants.append(missing)
        for value in variants:
            with self.subTest(value=value), \
                 patch('pathlib.Path.read_text', return_value=json.dumps(value)), \
                 self.assertRaises(ValueError):
                validate_noise_config(NOISY_CONFIG)

    def test_ground_truth_uses_cached_state_without_noisy_getter_calls(self):
        robot = SimpleNamespace(base_position=np.array([1., 2., 3.]),
            base_orientation=np.array([0., 0., 0., 1.]), motor_angles=np.arange(17.)/10.)
        sim = SimpleNamespace(bot=[robot], GetBasePosition=Mock(side_effect=AssertionError),
                              GetMotorAngles=Mock(side_effect=AssertionError))
        frame = SimpleNamespace(translation=np.array([.3, .4, .5]))
        dyn = SimpleNamespace(pin_model=object(), pin_data=SimpleNamespace(oMf=[frame]),
                              ReoderJoints2PinVec=Mock(side_effect=lambda state, kind:state))
        pin = SimpleNamespace(forwardKinematics=Mock(), updateFramePlacements=Mock())
        with patch.dict(sys.modules, {'pinocchio':pin}):
            ee, state = ground_truth_ee(sim, dyn, 0)
        np.testing.assert_array_equal(state, np.r_[robot.base_position,
                                                  robot.base_orientation, robot.motor_angles])
        np.testing.assert_array_equal(ee, [.3, .4, .5])
        sim.GetMotorAngles.assert_not_called()
        sim.GetBasePosition.assert_not_called()
        ee[:] = 99.
        self.assertNotEqual(frame.translation[0], 99.)

    def test_native_getter_standard_deviation_not_variance_and_no_cache_mutation(self):
        from simulation_and_control import pb
        sim = pb.SimInterface.__new__(pb.SimInterface)
        robot = SimpleNamespace(delay_measure_flag=False, noise_flag=True,
            motor_angles=np.zeros(17), robot_noise={'joint_cov':np.full(17, .001)})
        sim.bot = [robot]
        state = np.random.get_state()
        try:
            np.random.seed(31415)
            samples = np.asarray([sim.GetMotorAngles(0) for _ in range(5000)])
        finally:
            np.random.set_state(state)
        self.assertAlmostEqual(float(samples.std()), .001, delta=.00002)
        np.testing.assert_array_equal(robot.motor_angles, np.zeros(17))


if __name__ == '__main__':
    unittest.main()
