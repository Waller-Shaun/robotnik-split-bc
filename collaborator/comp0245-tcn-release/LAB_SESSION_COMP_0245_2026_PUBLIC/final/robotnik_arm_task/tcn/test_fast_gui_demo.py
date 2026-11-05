"""Display adapter tests, without opening windows or loading model weights."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import fast_gui_demo as module
from fast_gui_demo import DisplayedSimulation, LiveMirror, pack_pose


class DisplayAdapterTests(unittest.TestCase):
    def test_reads_and_configuration_writes_reach_physics(self):
        physics = SimpleNamespace(dt=.001, _SetMotorTorqueByIds=Mock())
        display = DisplayedSimulation(SimpleNamespace(physics=physics))
        self.assertEqual(display.dt, .001)
        replacement = Mock()
        display._SetMotorTorqueByIds = replacement
        self.assertIs(physics._SetMotorTorqueByIds, replacement)
        display._SetMotorTorqueByIds([1], [2.])
        replacement.assert_called_once_with([1], [2.])

    def test_step_delegates_once_and_returns_original_value(self):
        mirror = SimpleNamespace(physics=SimpleNamespace(), step=Mock(return_value=123))
        display = DisplayedSimulation(mirror)
        self.assertEqual(display.Step('command', 'torque'), 123)
        mirror.step.assert_called_once_with('command', 'torque')

    def test_pose_export_uses_only_cached_observations(self):
        robot = SimpleNamespace(base_position=[1, 2, 3], base_orientation=[0, 0, 0, 1],
                                motor_angles=[.1, .2], motor_velocities=[.3, .4])
        physics = SimpleNamespace(bot=[robot], pybullet_client=Mock())
        self.assertEqual(pack_pose(physics, 17),
                         [17., 1., 2., 3., 0., 0., 0., 1., .1, .2, .3, .4])
        self.assertEqual(physics.pybullet_client.mock_calls, [])

    def make_mirror(self):
        mirror = LiveMirror.__new__(LiveMirror)
        mirror.physics = SimpleNamespace(Step=Mock(return_value='stepped'))
        mirror.dt = .001
        mirror.steps = 0
        mirror.started = mirror.previous_step_end = None
        mirror.next_frame = 100.
        mirror.physics_s = mirror.display_s = mirror.sleep_s = mirror.between_steps_s = 0.
        mirror.realtime = True
        return mirror

    def test_no_sleep_when_computation_already_behind_deadline(self):
        mirror = self.make_mirror()
        with patch.object(module.time, 'perf_counter', side_effect=[1., 1.01, 1.01, 1.01, 1.01]), \
             patch.object(module.time, 'sleep') as sleep:
            self.assertEqual(mirror.step('command'), 'stepped')
        sleep.assert_not_called()
        mirror.physics.Step.assert_called_once_with('command')
        self.assertEqual(mirror.steps, 1)

    def test_pacing_waits_only_remaining_deadline_not_full_timestep(self):
        mirror = self.make_mirror()
        with patch.object(module.time, 'perf_counter',
                          side_effect=[1., 1.0002, 1.0002, 1.0002, 1.0002, 1.001, 1.001]), \
             patch.object(module.time, 'sleep') as sleep:
            mirror.step('command')
        self.assertAlmostEqual(sleep.call_args.args[0], .0008)
        mirror.physics.Step.assert_called_once_with('command')


if __name__ == '__main__':
    unittest.main()
