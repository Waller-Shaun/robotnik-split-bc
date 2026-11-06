"""Audit completed paired final trials and report target-cluster uncertainty."""
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from .common import ROOT,write,check_baseline
from .evaluate import files


def interval(values,heights):
    rng=np.random.default_rng(20261105)
    strata=[np.flatnonzero(heights==i) for i in range(10)]
    means=np.zeros(10000)
    for indices in strata:
        samples=rng.choice(indices,size=(10000,len(indices)),replace=True)
        means+=values[samples].sum(axis=1)/len(values)
    return (100*np.quantile(means,[.025,.975])).tolist()


def main():
    check_baseline()
    protocol=json.loads((ROOT/'protocol.json').read_text())
    selection=json.loads((ROOT/'selection.json').read_text())
    if files()!=selection['frozen_files']:
        raise ValueError('Selected candidate changed')
    output=ROOT/'outputs/final'
    status=json.loads((output/'summary.json').read_text())
    rows=json.loads((output/'results.json').read_text())
    if status['state']!='completed' or len(rows)!=3000:
        raise ValueError('Final test not complete; do not publish partial success rate')
    pairs={}
    gain_ranges=[]
    for row in rows:
        key=(row['variant'],row['target_index'],row['repeat_index'])
        if key in pairs:raise ValueError('Duplicated trial')
        pairs[key]=row
        entry=protocol['final'][row['target_index']]
        if row['seed']!=entry['noise_seeds'][row['repeat_index']]:
            raise ValueError('Noise seed differs from declared final protocol')
        if row['status']=='process_error':continue
        path=Path(row['rollout_file'])
        with np.load(path,allow_pickle=False) as d:
            np.testing.assert_array_equal(d['target'],entry['target'])
            error=np.linalg.norm(d['ee_positions']-d['target'],axis=1)
            np.testing.assert_allclose(error,d['ee_errors'],atol=1e-14,rtol=0)
            assert bool((error<.01).any())==row['success']
            assert abs(error.min()-row['min_error_m'])<1e-12
        if row['variant']=='dynamic':
            with np.load(path.with_suffix('.gains.npz'),allow_pickle=False) as g:
                a=g['trace']
                if len(a):
                    assert a.shape[1]==29
                    assert (a[:,1:8]>=80-1e-5).all() and (a[:,1:8]<=120+1e-5).all()
                    assert (a[:,8:15]>=24-1e-5).all() and (a[:,8:15]<=36+1e-5).all()
            gain_ranges.append(row['gain_summary'])
    scores={v:np.array([[pairs[v,i,j]['success'] for j in range(3)] for i in range(500)],float) for v in ('fixed','dynamic')}
    heights=np.array([e['height_bin'] for e in protocol['final']])
    metrics={}
    for variant,values in scores.items():
        subset=[r for r in rows if r['variant']==variant]
        failures=[r for r in subset if not r['success']]
        metrics[variant]=dict(successes=int(values.sum()),episodes=1500,distinct_targets=500,
            success_rate_percent=float(100*values.mean()),
            target_cluster_stratified_bootstrap_95_percent=interval(values.mean(axis=1),heights),
            success_by_repeat=values.sum(axis=0).astype(int).tolist(),
            all_three_success_targets=int((values.sum(axis=1)==3).sum()),
            process_errors=sum(r['status']=='process_error' for r in subset),
            navigation_incomplete_failures=sum(r['status']!='process_error' and r['phase_switch_time_s'] is None for r in failures),
            mean_success_time_s=float(np.mean([r['end_time_s'] for r in subset if r['success']])),
            mean_clipped_fraction=float(np.mean([r['clipped_fraction'] for r in subset if r['status']!='process_error'])),
            failures=[{k:r.get(k) for k in ('target_index','repeat_index','status','min_error_m','final_error_m','phase_switch_time_s')} for r in failures])
    difference=scores['dynamic']-scores['fixed']
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),final_test_complete=True,
        raw_trajectory_audit_passed=True,original_baseline_and_submission_unchanged=True,
        candidate_frozen_before_test=True,metrics=metrics,
        paired_difference_percentage_points=float(100*difference.mean()),
        paired_target_cluster_95_percent=interval(difference.mean(axis=1),heights),
        paired_improvements=int((difference>0).sum()),paired_regressions=int((difference<0).sum()),
        gain_min=np.min([g['minimum'] for g in gain_ranges],axis=0).tolist(),
        gain_max=np.max([g['maximum'] for g in gain_ranges],axis=0).tolist(),
        mean_within_episode_gain_std=np.mean([g['std'] for g in gain_ranges],axis=0).tolist(),
        is_structured_hybrid_not_unrestricted_torque_TCN=True,
        instructor_acceptance_not_confirmed=True,old_99p07_not_new_model_score=True)
    write(ROOT/'FINAL_RESULTS.json',summary)
    f,d=metrics['fixed'],metrics['dynamic']
    text=f'''# State-dependent P/D: final independent paired experiment

| Model | New targets | Noise repeats | Success / trials | Success rate |
|---|---:|---:|---:|---:|
| Fixed P=100, D=30 | 500 | 3 | {f['successes']}/1500 | {f['success_rate_percent']:.2f}% |
| Learned state-dependent P/D | 500 | 3 | {d['successes']}/1500 | {d['success_rate_percent']:.2f}% |

Dynamic minus fixed: {summary['paired_difference_percentage_points']:.3f} percentage points;
target-cluster stratified bootstrap 95% interval: {summary['paired_target_cluster_95_percent']}.
Paired rescues: {summary['paired_improvements']}; paired regressions: {summary['paired_regressions']}.
Process errors: fixed {f['process_errors']}, dynamic {d['process_errors']} (included in denominators).

This is the originally unused reserve: 500 new goals x3 predeclared noise seeds,
not 1500 independent goals. Shared fixed initial pose, 20 s simulation horizon,
1 ms physics/control, true EE error <1 cm anytime, no hold requirement. Same
native eight noise standard deviations 0.001, same simulator/contact settings.
All histories, integrated references and learned-gain filter state reset each
episode. All full-rate true scoring trajectories/actions were audited; large
diagnostic sensor/robot-state arrays were stored at 10 ms, not at every tick.
No test trajectory was used
to train or select the gain head. No final failures were removed.

## What was learned

A 128->32->14 neural head predicts bounded P in 80..120 and D in 24..36 from
frozen causal TCN features. Training used expert torque supervision on the
original 698 train /144 validation split, with deviation and temporal
regularization. Motion/gravity/navigation networks stayed frozen. A 10 ms
gain filter was applied online. Exact source, model and configuration hashes
were frozen before final outcomes. `FINAL_RESULTS.json` reports per-joint
gain ranges/variation and every final failure.

## What this does NOT establish

This is still a disclosed structured/hybrid controller with a PD torque map
and reference integrator; learning gains is not proof of instructor acceptance.
The coursework clearly requests supervised state/target->robot-command
regression; this implementation trains on expert final torques and its full
predictor returns wheel commands and torques, but its structured formulation
should be explicitly described and confirmed if the instructor requires an
unrestricted direct-torque regressor. No stability theorem or real-robot
safety guarantee is claimed. Bounds, filtering and tests are empirical checks.

Expert gains were constant. Learned P/D are compensatory policy parameters,
not uniquely identified physical gains. Offline expert reference positions
are used for the training loss only, whereas online references are integrated
from learned velocity; this teacher-forcing gap remains. Gain uncertainty,
multiple training seeds, arbitrary initial poses, obstacles and external
pushes were not comprehensively evaluated. Paper inspiration is not a full
reproduction of SCAPE/VICES, which use different tasks/learning methods.

The old 99.07% is a different target set's historical result, not the score
of this dynamic-gain model. The old ZIP/GitHub release was not overwritten.
Do not claim dynamic gains improve performance if the paired confidence
interval includes zero. Do not silently adopt the candidate if it regresses.
'''
    (ROOT/'EXPERIMENT_REPORT.md').write_text(text)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('metrics',)},indent=2))
    print('FINAL: '+json.dumps({v:{k:m[k] for k in ('successes','episodes','success_rate_percent','target_cluster_stratified_bootstrap_95_percent')} for v,m in metrics.items()}),flush=True)


if __name__=='__main__':main()
