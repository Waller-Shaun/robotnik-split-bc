"""Verify files/syntax and optionally compare frozen MLP forwards; no simulation/training."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-check',action='store_true',help='Load torch checkpoints and compare original class forward outputs')
    args=parser.parse_args()
    folder=ROOT/'artifacts/patrol120'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for rel,digest in manifest['files'].items():
        if hashlib.sha256((folder/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Checksum mismatch: '+rel)
    files=list((ROOT/'patrol').glob('*.py'))
    for file in files:ast.parse(file.read_text(encoding='utf-8'),filename=str(file))
    result={'checksums_passed':len(manifest['files']),'python_syntax_passed':len(files),
            'simulation_started':False,'training_started':False,'model_forward_check_performed':False}
    if args.model_check:
        import torch
        from torch import nn
        from patrol_checkpoint_bundle import load_patrol_bundle
        torch.set_num_threads(1)
        arm,nav,*_=load_patrol_bundle(folder)
        refroot=ROOT/'research/source/LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task'
        gen=torch.Generator().manual_seed(73)
        for model,name,classname,checkpoint,width in [
            (arm,'robotnik_arm_aux_policy.py','AuxiliaryArmPolicy',folder/'arm120/best_model.pt',26),
            (nav,'robotnik_navigation_policy.py','NavigationPolicy',folder/'navigation/best_model.pt',8)]:
            tree=ast.parse((refroot/name).read_text(encoding='utf-8'))
            node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
            namespace={'torch':torch,'nn':nn}
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<original_policy_class>','exec'),namespace)
            reference=namespace[classname]().eval()
            reference.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state_dict'],strict=True)
            x=torch.randn(32,width,generator=gen)
            with torch.inference_mode():torch.testing.assert_close(model(x),reference(x),rtol=0,atol=0)
        result.update(model_forward_check_performed=True,original_class_outputs_bitwise_equal=True)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
