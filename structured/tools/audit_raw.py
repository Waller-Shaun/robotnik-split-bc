"""Independent local raw-archive audit, including FK and actuator clipping."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_experiment import verify
from dynamic_pd.common import BASE, write


def main():
    verify(runtime=True)
    output = ROOT/'outputs/final'
    status = json.loads((output/'summary.json').read_text())
    rows = json.loads((output/'results.json').read_text())
    if status['state'] != 'completed' or len(rows) != 3000:
        raise ValueError('Wait for the complete final test')
    import pinocchio as pin
    from simulation_and_control import pb, PinWrapper
    config = BASE/'artifacts/RoboEnv/configs/robotnik_arm_noise_0p001.json'
    assets = BASE/'external/RoboEnv'
    sim = pb.SimInterface(str(config), conf_file_path_ext=str(assets), use_gui=False)
    try:
        names = np.expand_dims(np.asarray(sim.getNameActiveJoints()), 0)
        dyn = PinWrapper(str(config), 'pybullet', names, ['pybullet'], False, 0, str(assets))
    finally:
        sim.pybullet_client.disconnect()
    frame = dyn.pin_model.getFrameId('robot_j2s7s300_end_effector')
    limits = np.array([3.]*4+[30.]*7, dtype=np.float32)
    checked = fk_samples = 0
    max_fk_error = 0.
    totals = {v:dict(action_steps=0, inference_ms_sum=0., trial_means=[]) for v in ('fixed', 'dynamic')}
    for index, row in enumerate(rows, 1):
        if row['status'] == 'process_error':
            continue
        with np.load(row['rollout_file'], allow_pickle=False) as d:
            actions, raw = d['actions'], d['raw_actions']
            times, positions, target = d['timestamps'], d['ee_positions'], d['target']
            errors = np.linalg.norm(positions-target, axis=1)
            assert len(times) == len(actions)+1
            assert raw.shape == actions.shape and actions.shape[1] == 11
            assert np.isfinite(actions).all() and np.isfinite(raw).all()
            np.testing.assert_array_equal(actions, np.clip(raw, -limits, limits))
            np.testing.assert_allclose(times, np.arange(len(times))*.001, rtol=0, atol=1e-13)
            np.testing.assert_array_equal(d['action_timestamps'], times[:-1])
            np.testing.assert_allclose(errors, d['ee_errors'], rtol=0, atol=1e-14)
            assert bool((errors < .01).any()) == row['success'] == bool(d['success'])
            assert abs(errors[-1]-row['final_error_m']) < 1e-12
            assert abs(errors.min()-row['min_error_m']) < 1e-12
            summary = json.loads(str(d['summary_json']))
            assert summary['sensor_noise_enabled'] == [True]
            assert summary['ground_truth_exposed_to_policy'] is False
            assert summary['reach_tolerance_m'] == .01 and summary['required_hold_s'] == 0
            assert summary['duration_limit_s'] == 20 and times[-1] <= 20
            assert summary['evaluation_seed'] == row['seed']
            samples, states = d['diagnostic_sample_indices'], d['ground_truth_states_10ms']
            for i in np.unique([0, len(samples)//2, len(samples)-1]):
                q_pin = dyn.ReoderJoints2PinVec(states[i], 'pos')
                pin.forwardKinematics(dyn.pin_model, dyn.pin_data, q_pin)
                pin.updateFramePlacements(dyn.pin_model, dyn.pin_data)
                error = float(np.linalg.norm(dyn.pin_data.oMf[frame].translation-positions[samples[i]]))
                assert error < 1e-10
                max_fk_error = max(max_fk_error, error)
                fk_samples += 1
            inference = d['inference_ms']
            assert len(inference) == len(actions) and np.isfinite(inference).all()
            total = totals[row['variant']]
            total['action_steps'] += len(actions)
            total['inference_ms_sum'] += float(inference.sum())
            total['trial_means'].append(float(inference.mean()))
        checked += 1
        if index%250 == 0:
            print(f'Raw audit: {index}/3000 rows', flush=True)
    runtime = {v:dict(action_steps=t['action_steps'],
        mean_per_control_step_inference_ms=t['inference_ms_sum']/t['action_steps'],
        mean_of_trial_mean_inference_ms=float(np.mean(t['trial_means']))) for v,t in totals.items()}
    result = dict(passes=True, completed_trial_records=3000, raw_archives_checked=checked,
        process_errors_retained=status['process_errors'], forward_kinematics_samples=fk_samples,
        maximum_true_fk_disagreement_m=max_fk_error, full_rate_action_clipping_verified=True,
        full_rate_true_error_and_success_verified=True, timestamps_and_noise_metadata_verified=True,
        runtime=runtime, timing_scope='CPU inference under concurrent benchmark load, not an isolated latency guarantee')
    write(ROOT/'RAW_ARCHIVE_AUDIT.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
