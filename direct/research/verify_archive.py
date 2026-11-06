"""Static archive verification only: no Torch, simulation, training or network."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def main():
    manifest=json.loads((ROOT/'research/archive_manifest.json').read_text(encoding='utf-8'))
    failures=[]
    for item in manifest['files']:
        path=ROOT/item['path']
        if not path.is_file() or sha(path)!=item['sha256']:
            failures.append(item['path'])
        if path.is_file() and path.stat().st_size>=100*1024**2:
            failures.append('GitHub file limit: '+item['path'])
    indexes=json.loads((ROOT/'research/model_index.json').read_text(encoding='utf-8'))
    count=sum(len(e['checkpoints']) for e in indexes)
    assert len(indexes)==manifest['source_runs']==34
    assert count==manifest['checkpoint_files']==61
    team=ROOT/'collaborator/comp0245-tcn-release'
    original=json.loads((team/'release_manifest.json').read_text(encoding='utf-8'))
    for rel,expected in original['file_sha256'].items():
        path=team/rel
        if not path.is_file() or sha(path)!=expected:
            failures.append('Teammate original manifest: '+rel)
    for condition,expected_n,expected_success in [('friend',1500,1486),('off',500,493)]:
        rows=json.loads((team/f'results/recorded_primary/{condition}/results.json').read_text(encoding='utf-8'))
        assert len(rows)==expected_n and sum(bool(x['success']) for x in rows)==expected_success
        assert len({x['target_index'] for x in rows})==500
    recovery=ROOT/'research/data/expert_recovery_data/collection_20261104_112547_770894/episodes'
    data=json.loads((recovery/'recovery_manifest.json').read_text(encoding='utf-8'))
    assert data['trajectories']==102 and data['frames']==233150
    for e in data['entries']:
        if sha(recovery/e['file'])!=e['sha256']:
            failures.append('Recovery: '+e['file'])
    out={'status':'passed' if not failures else 'failed','files_verified':len(manifest['files']),
         'our_runs':len(indexes),'our_checkpoints':count,'teammate_manifest_verified':not any('Teammate' in x for x in failures),
         'recovery_episodes':102,'training_or_simulation_executed':False,'failures':failures}
    print(json.dumps(out,ensure_ascii=False,indent=2))
    if failures: raise SystemExit(1)
if __name__=='__main__':main()
