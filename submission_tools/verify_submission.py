"""Read-only hashes and syntax checks. No imports of experiment code or Torch."""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    manifest=json.loads((ROOT/'SUBMISSION_MANIFEST.json').read_text(encoding='utf-8'))
    counts={'files':0,'python_syntax':0,'json_parsed':0}
    failures=[]
    for item in manifest['files']:
        path=ROOT/item['path']
        if not path.is_file() or sha(path)!=item['sha256']:
            failures.append('Missing/changed: '+item['path']);continue
        counts['files']+=1
        try:
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8-sig'));counts['python_syntax']+=1
            elif path.suffix=='.json':json.loads(path.read_text(encoding='utf-8-sig'));counts['json_parsed']+=1
        except Exception as exc:failures.append(item['path']+': '+type(exc).__name__+': '+str(exc))
    inventory=json.loads((ROOT/'SOURCE_INVENTORY.json').read_text(encoding='utf-8'))
    for item in inventory['files']:
        if not (ROOT/item['path']).is_file() or sha(ROOT/item['path'])!=item['sha256']:
            failures.append('Original source differs: '+item['path'])
    print(json.dumps({'status':'passed' if not failures else 'failed',**counts,'original_source_files':len(inventory['files']),
        'training_executed':False,'simulation_executed':False,'network_accessed':False,'failures':failures},indent=2))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
