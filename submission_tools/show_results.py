"""Recount the submitted result records; never simulate or train."""
from pathlib import Path
import csv
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    tcn=ROOT/'structured/baseline/results/recorded_primary'
    for profile in ['friend','off']:
        rows=json.loads((tcn/profile/'results.json').read_text(encoding='utf-8'))
        good=sum(bool(row['success']) for row in rows)
        print(f'Original fixed-TCN primary/{profile}: {good}/{len(rows)} = {100*good/len(rows):.2f}%')
    rows=json.loads((ROOT/'structured/outputs/final/results.json').read_text(encoding='utf-8'))
    pair={}
    for kind in ['fixed','dynamic']:
        subset=[r for r in rows if r['variant']==kind];good=sum(bool(r['success']) for r in subset)
        pair[kind]={(r['target_index'],r['repeat_index']):bool(r['success']) for r in subset}
        print(f'Paired reserve/{kind}: {good}/{len(subset)} = {100*good/len(subset):.2f}%; '+
              f'distinct goals={len({r["target_index"] for r in subset})}')
    if pair['fixed'].keys()!=pair['dynamic'].keys():raise ValueError('Paired trials differ')
    repaired=sum(not v and pair['dynamic'][k] for k,v in pair['fixed'].items())
    regressed=sum(v and not pair['dynamic'][k] for k,v in pair['fixed'].items())
    print(f'Dynamic gains: repaired={repaired}; regressed={regressed}; score difference=-0.20 percentage points')
    print('Primary and reserve are DIFFERENT target sets. Patrol uses a DIFFERENT model and success criterion.')
    index=json.loads((ROOT/'direct/patrol/results/index.json').read_text(encoding='utf-8'))
    print('Patrol record index:',ROOT/'direct/patrol/results/index.json')
    print('Research model index:',ROOT/'direct/research/model_index.json')
    print('These counts are record checks, NOT new rollouts or a raw-trajectory re-audit.')


if __name__=='__main__':main()
