"""Submission dispatcher checks; no child process, model or simulator runs."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('submission_dispatch',ROOT/'submission.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DispatchChecks(unittest.TestCase):
    def test_fixed_and_dynamic_share_environment_layout(self):
        cwd,args=module.command_for('fixed-tcn',['--headless'])
        self.assertEqual(cwd,ROOT/'structured/baseline');self.assertEqual(args[:3],['run_model.py','--model','tcn'])
        cwd,args=module.command_for('dynamic-pd',['--headless'])
        self.assertEqual(cwd,ROOT/'structured');self.assertEqual(args[:2],['-m','dynamic_pd.demo'])

    def test_patrol_supplies_default_asset_root_and_honors_override(self):
        _,args=module.command_for('patrol',['--num-targets','5'])
        self.assertIn(str(ROOT/'structured/baseline/external/RoboEnv'),args)
        _,args=module.command_for('patrol',['--roboenv-root=MY_ROOT'])
        self.assertNotIn('--roboenv-root',args)

    def test_retraining_and_evaluation_are_separate(self):
        _,train=module.command_for('retrain-gain',[])
        _,test=module.command_for('replay',[])
        self.assertEqual(train,['tools/retrain_gain.py']);self.assertEqual(test,['tools/replay_benchmark.py'])

    def test_relative_paths_resolve_before_changing_child_directory(self):
        args=module.resolve_forwarded_paths('retrain-gain',['--data-root','dataset','--output=run_new'],ROOT)
        self.assertEqual(args,['--data-root',str((ROOT/'dataset').resolve()),'--output='+str((ROOT/'run_new').resolve())])

    def test_target_coordinates_are_never_rewritten(self):
        args=['--target','-0.9','1.2','1.1','--headless']
        self.assertEqual(module.resolve_forwarded_paths('fixed-tcn',args,ROOT),args)


if __name__=='__main__':unittest.main()
