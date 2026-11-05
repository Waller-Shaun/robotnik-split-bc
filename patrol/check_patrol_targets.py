"""Sampling/argument checks only; never import or start the simulation entry point."""
import argparse
import ast
from pathlib import Path
import secrets
import sys
import contextlib
import io
import unittest
from unittest.mock import patch
import numpy as np
import patrol_target_sampling as sampling


def parse_options(options):
    # 只执行入口中的参数解析函数，不导入PyBullet/Torch，也不执行main。
    tree=ast.parse(Path(__file__).with_name('test_patrol_120.py').read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_args')
    namespace={**vars(sampling),'argparse':argparse,'Path':Path,'np':np,'secrets':secrets,'__doc__':'CLI check','__file__':str(Path(__file__).with_name('test_patrol_120.py'))}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<parse_args_only>','exec'),namespace)
    with patch.object(sys,'argv',['test_patrol_120.py',*options]):
        return namespace['parse_args']()


class RandomTargetChecks(unittest.TestCase):
    def test_range_separation_and_noncollinearity_for_many_seeds(self):
        for seed in range(100):
            t=sampling.sample_three_targets(seed)
            self.assertEqual(t.shape,(3,3))
            r=np.linalg.norm(t[:,:2],axis=1)
            self.assertTrue(np.all((r>=.8)&(r<=2.)))
            self.assertTrue(np.all((t[:,2]>=.5)&(t[:,2]<=1.3)))
            self.assertGreaterEqual(min(np.linalg.norm(t[i,:2]-t[j,:2])
                                        for i,j in ((0,1),(0,2),(1,2))),.4)
            a,b=t[1,:2]-t[0,:2],t[2,:2]-t[0,:2]
            self.assertGreaterEqual(.5*abs(a[0]*b[1]-a[1]*b[0]),.1)

    def test_seed_reproduces_coordinates(self):
        np.testing.assert_array_equal(sampling.sample_three_targets(73),sampling.sample_three_targets(73))
        self.assertFalse(np.array_equal(sampling.sample_three_targets(73),sampling.sample_three_targets(74)))

    def test_global_noise_rng_is_not_consumed(self):
        before=np.random.get_state()
        sampling.sample_three_targets(73)
        after=np.random.get_state()
        self.assertEqual(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1])
        self.assertEqual(before[2:],after[2:])

    def test_default_cli_samples_three_targets_and_keeps_route_sorting(self):
        with patch.object(secrets,'randbits',return_value=73):
            args=parse_options([])
        self.assertEqual(args.target_selection['mode'],'random')
        self.assertEqual(args.target_selection['target_seed'],73)
        self.assertEqual(args.route_order,'shortest')
        self.assertEqual(args.seed,42)
        np.testing.assert_array_equal(args.targets,sampling.sample_three_targets(73))

    def test_cli_target_seed_reproduces_targets(self):
        args=parse_options(['--target-seed','73'])
        np.testing.assert_array_equal(args.targets,sampling.sample_three_targets(73))
        self.assertEqual(args.target_selection['target_seed_source'],'argument')

    def test_explicit_target_bypasses_random_sampling(self):
        args=parse_options(['--target','0','1.2','1','--target','.4','1.2','.75','--target-seed','73'])
        np.testing.assert_array_equal(args.targets,[[0.,1.2,1.],[.4,1.2,.75]])
        self.assertEqual(args.target_selection['mode'],'explicit')
        self.assertFalse(args.target_selection['random_targets_generated'])

    def test_custom_count_ranges_spacing_and_fixed_height(self):
        args=parse_options(['--num-targets','5','--radius-range','1','1.8',
                            '--height-range','.75','.75','--min-target-spacing','.3','--target-seed','73'])
        self.assertEqual(args.targets.shape,(5,3))
        r=np.linalg.norm(args.targets[:,:2],axis=1)
        self.assertTrue(np.all((r>=1.)&(r<=1.8)))
        np.testing.assert_array_equal(args.targets[:,2],np.full(5,.75))
        self.assertGreaterEqual(min(np.linalg.norm(args.targets[i,:2]-args.targets[j,:2])
                                   for i in range(5) for j in range(i+1,5)),.3)
        self.assertTrue(args.show_planned_route)

    def test_single_and_two_targets_do_not_require_triangle_area(self):
        for count in (1,2):
            args=parse_options(['--num-targets',str(count),'--target-seed','73'])
            self.assertEqual(args.targets.shape,(count,3))
            self.assertIsNone(args.target_selection['minimum_triangle_xy_area_m2'])

    def test_expanded_radius_is_recorded_and_display_can_be_disabled(self):
        args=parse_options(['--num-targets','3','--radius-range','2.5','3',
                            '--target-seed','73','--no-show-planned-route'])
        self.assertFalse(args.show_planned_route)
        self.assertEqual(args.target_selection['initial_target_ids_outside_training_radius'],[1,2,3])

    def test_invalid_parameter_combinations_fail_before_simulation(self):
        cases=[['--num-targets','0'],['--height-range','.4','1.'],
               ['--height-range','1.','1.4'],['--radius-range','2','1'],
               ['--min-target-spacing','-1'],['--radius-range','.8','1','--min-target-spacing','3']]
        for options in cases:
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                parse_options(options)
            self.assertEqual(error.exception.code,2)


if __name__=='__main__':unittest.main()
