"""Read expert data only; cache exact-window frozen TCN features, not test data."""
import json
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, BASE, sha, write, load_fixed
from robotnik_bodyframe_features import build_bodyframe_observations
from structured_arm import features, decode


def prepare(split, smoke, device):
    if split not in ('train','val'):
        raise ValueError('Only original train/val may be loaded for gain learning')
    manifest = json.loads((BASE / 'models/tcn/data_split.json').read_text())
    filenames = manifest['splits'][split]
    if smoke:
        heights = manifest['targets'][split]
        chosen = []
        for low in (True,False):
            chosen.extend([i for i,t in enumerate(heights) if (t[2]<=.5)==low][:2])
        filenames = [filenames[i] for i in chosen]
    local = json.loads((ROOT / 'local_paths.json').read_text())
    data_root = Path(local['dataset_root']) / 'RoboEnv'
    cache = ROOT / 'cache' / ('smoke_' if smoke else 'full_')
    cache = cache.with_name(cache.name + split + '.npz')
    metadata = dict(split=split,filenames=filenames,source_manifest_sha256=sha(BASE/'models/tcn/data_split.json'),
                    frozen_model_sha256=sha(BASE/'models/tcn/best_model.pt'),
                    feature_source_sha256=sha(Path(__file__)),test_loaded=False)
    if cache.exists():
        if json.loads(cache.with_suffix('.json').read_text()) != metadata:
            raise ValueError('Stale/changed feature cache; use a new experiment')
        print('Reusing verified feature cache: ' + str(cache), flush=True)
        with np.load(cache,allow_pickle=False) as stored:
            return {key:stored[key].copy() for key in stored.files}, filenames
    controller = load_fixed()
    model = controller.model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    gm,gs = controller.g_mean.to(device),controller.g_std.to(device)
    arrays = {key:[] for key in ('hidden','velocity','gravity','q_error','qd','torque','low','weight','previous')}
    total = 0
    for number,filename in enumerate(filenames,1):
        with np.load(data_root/filename,allow_pickle=False) as d:
            indices = np.arange(0,len(d['observations']),10)
            obs = d['observations'][indices].astype(np.float32)
            raw = features(build_bodyframe_observations(obs,d['ee_positions'][indices]),d['base_orientations'][indices])
            normalized = (raw-controller.stats['x_mean'])/controller.stats['x_std']
            hidden,velocities,gravities = [],[],[]
            with torch.inference_mode():
                for begin in range(0,len(raw),1024):
                    rows = np.arange(begin,min(begin+1024,len(raw)))
                    # Parent network was trained on exactly 20 frames. A full
                    # trajectory convolution would accidentally use 29 frames.
                    windows = normalized[np.maximum(rows[:,None]-np.arange(19,-1,-1)[None],0)]
                    x = torch.from_numpy(np.ascontiguousarray(windows)).to(device)
                    h = model.encoder(x.transpose(1,2))[:,:,-1]
                    v,g = decode(model.head(h),torch.from_numpy(raw[rows]).to(device),gm,gs)
                    hidden.append(h.cpu().numpy().copy())
                    velocities.append(v.cpu().numpy().copy())
                    gravities.append(g.cpu().numpy().copy())
            values = dict(hidden=np.concatenate(hidden),velocity=np.concatenate(velocities),
                gravity=np.concatenate(gravities),q_error=d['q_des_arm'][indices].astype(np.float32)-obs[:,5:12],
                qd=obs[:,12:19],torque=d['actions'][indices,4:11].astype(np.float32),
                low=np.full(len(raw),bool(d['target'][2]<=.5)),
                weight=np.where((raw[:,24]==0)&(np.linalg.norm(raw[:,25:28],axis=1)<.1),3.,1.),
                previous=total+np.maximum(np.arange(len(raw))-1,0))
            if not all(np.isfinite(v).all() for v in values.values()):
                raise ValueError('Nonfinite supervised data')
            for key,value in values.items():
                arrays[key].append(value)
            total += len(raw)
        if number%50==0 or number==len(filenames):
            print(f'{split}: cached {number}/{len(filenames)} trajectories; {total} frames',flush=True)
    arrays = {key:np.concatenate(value) for key,value in arrays.items()}
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache,**arrays)
    write(cache.with_suffix('.json'),metadata)
    return arrays,filenames
