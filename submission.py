"""Explicit entry points for the COMP0245 assignment submission.

No training or simulation starts when this module is imported or with --help.
Only explicit demo/replay/retrain commands dispatch those operations.
"""
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
DIRECT=ROOT/'direct'
STRUCTURED=ROOT/'structured'
ROBOENV=STRUCTURED/'baseline/external/RoboEnv'

HELP='''COMP0245 code submission

Run in the pinned ML environment after following README.md.

Read-only checks (standard Python, no simulator required):
  python -X utf8 submission.py verify
  python -X utf8 submission.py results

Explicit simulation commands (ONLY when you choose to run them):
  python -X utf8 submission.py fixed-tcn --target -0.9 1.2 1.1 --seed 137
  python -X utf8 submission.py dynamic-pd --target -0.9 1.2 1.1 --seed 137
  python -X utf8 submission.py patrol --num-targets 5 --target-seed 1266108476
  python -X utf8 submission.py replay --workers 4 --limit 2

Explicit data-dependent training (not needed to inspect or run the models):
  python -X utf8 submission.py retrain-gain --data-root PATH --output NEW_DIRECTORY

fixed-tcn and dynamic-pd use the SAME pinned structured/baseline/external/RoboEnv.
patrol uses that asset root by default; --roboenv-root PATH overrides it.
Training is separate from testing. Historical experiments remain archived.
'''


def resolve_forwarded_paths(command,args,caller):
    """Keep explicit user paths relative to the caller, not the child cwd."""
    flags={'fixed-tcn':{'--result-file'},'patrol':{'--roboenv-root','--robot-config','--output-dir'},
           'retrain-gain':{'--data-root','--output'}}.get(command,set())
    result=[];i=0
    while i<len(args):
        word=args[i]
        if word in flags:
            if i+1>=len(args):raise ValueError('Missing path after '+word)
            result.extend([word,str((caller/Path(args[i+1]).expanduser()).resolve())]);i+=2
        elif '=' in word and word.split('=',1)[0] in flags:
            key,value=word.split('=',1)
            if not value:raise ValueError('Empty path after '+key)
            result.append(key+'='+str((caller/Path(value).expanduser()).resolve()));i+=1
        else:result.append(word);i+=1
    return result


def command_for(command,args):
    if command=='verify':return ROOT,[str(ROOT/'submission_tools/verify_submission.py'),*args]
    if command=='results':return ROOT,[str(ROOT/'submission_tools/show_results.py'),*args]
    if command=='fixed-tcn':return STRUCTURED/'baseline',['run_model.py','--model','tcn',*args]
    if command=='dynamic-pd':return STRUCTURED,['-m','dynamic_pd.demo',*args]
    if command=='replay':return STRUCTURED,['tools/replay_benchmark.py',*args]
    if command=='retrain-gain':return STRUCTURED,['tools/retrain_gain.py',*args]
    if command=='patrol':
        extra=[] if any(a=='--roboenv-root' or a.startswith('--roboenv-root=') for a in args) else ['--roboenv-root',str(ROBOENV)]
        return DIRECT,['patrol/test_patrol_120.py',*extra,*args]
    raise ValueError('Unknown command: '+command)


def main():
    if len(sys.argv)<2 or sys.argv[1] in ['-h','--help']:
        print(HELP);return
    try:
        forwarded=resolve_forwarded_paths(sys.argv[1],sys.argv[2:],Path.cwd())
        cwd,arguments=command_for(sys.argv[1],forwarded)
    except ValueError as exc:print(exc,file=sys.stderr);raise SystemExit(2)
    env=os.environ.copy()
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
    if sys.argv[1] not in ['verify','results']:
        for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env.setdefault(key,'1')
        print('Explicit command:',sys.argv[1],'; working directory:',cwd,flush=True)
    result=subprocess.run([sys.executable,'-X','utf8','-B',*arguments],cwd=cwd,env=env)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
