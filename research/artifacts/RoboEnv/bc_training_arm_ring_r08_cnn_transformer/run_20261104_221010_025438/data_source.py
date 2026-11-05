"""[CNN-T data] 每帧只存一次，批量按需生成历史窗口，保留轨迹/恢复片段边界。"""
from pathlib import Path
import numpy as np
import torch
from train_arm_bc_ring_aux_b2 import check_headers
from robotnik_bodyframe_features import build_bodyframe_observations
from recovery_residual120_context import load_recovery


def validate_times(times,n):
    if times.shape!=(n,) or n<2 or not np.isfinite(times).all():raise ValueError('Invalid timestamps')
    dt=float(np.median(np.diff(times)))
    if abs(dt-.001)>1e-9 or not np.allclose(np.diff(times),dt,atol=1e-9,rtol=0):
        raise ValueError('Expected original 1ms aligned trajectories')


def original_arrays(manifest,split,stats,limit=None):
    files,lengths=check_headers(manifest,split,limit)
    n=sum(lengths)
    x=np.empty((n,26),np.float32);y=np.empty((n,7),np.float32);low=np.empty(n,bool)
    offset=0
    for count,((filename,target),length) in enumerate(zip(files,lengths),1):
        with np.load(Path(manifest['data_dir'])/filename,allow_pickle=False) as d:
            obs=d['observations'];action=d['actions']
            validate_times(d['timestamps'],length)
            if not np.isfinite(obs).all() or not np.isfinite(action).all() or not np.isin(obs[:,22],[0,1]).all():
                raise ValueError('Invalid original data: '+filename)
            np.testing.assert_allclose(d['target'],target,atol=1e-6,rtol=0)
            sl=slice(offset,offset+length)
            x[sl]=(build_bodyframe_observations(obs,d['ee_positions'])-stats['x_mean'])/stats['x_std']
            y[sl]=(action[:,4:11]-stats['y_mean'][4:11])/stats['y_std'][4:11]
            low[sl]=target[2]<=.35+1e-6
        offset+=length
        if count%100==0 or count==len(files):print(f'{split}: {count}/{len(files)} original trajectories',flush=True)
    if not low.any() or low.all():raise ValueError('Both height groups required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite normalized data')
    return x,y,low,np.asarray(lengths,np.int64)


def recovery_arrays(path,parent,split,stats,limit=None):
    # 复用已有严格核验：同一113采集来源、相同120数据上下文、train归属和每文件SHA256。
    x,y,low,manifest=load_recovery(path,parent,split,stats)
    entries=manifest['entries'] if limit is None else manifest['entries'][:limit]
    lengths=np.array([e['frames'] for e in entries],dtype=np.int64)
    for e in entries:
        with np.load(Path(path).parent/e['file'],allow_pickle=False) as d:validate_times(d['timestamps'],e['frames'])
    count=int(lengths.sum())
    return x[:count],y[:count],low[:count],lengths,manifest


class WindowStore:
    """输入可以是多个轨迹块；恢复片段不与原专家轨迹拼接成连续时间。"""
    def __init__(self,parts,device,history_length=16,stride=10):
        counts=[len(p[0]) for p in parts];n=sum(counts)
        self.x=torch.empty((n,26),dtype=torch.float32,device=device)
        self.y=torch.empty((n,7),dtype=torch.float32,device=device)
        self.low=torch.empty(n,dtype=torch.bool,device=device)
        lengths=[];offset=0
        for (x,y,low,lens),count in zip(parts,counts):
            if sum(lens)!=count or np.any(lens<=0):raise ValueError('Invalid sequence lengths')
            self.x[offset:offset+count]=torch.from_numpy(x).to(device)
            self.y[offset:offset+count]=torch.from_numpy(y).to(device)
            self.low[offset:offset+count]=torch.from_numpy(low).to(device)
            lengths.extend(lens.tolist());offset+=count
        cumulative=np.cumsum(lengths)
        self.ends=torch.as_tensor(cumulative,device=device)
        self.starts=torch.as_tensor(np.r_[0,cumulative[:-1]],device=device)
        self.offsets=torch.arange(history_length-1,-1,-1,device=device)*stride
        self.n=n

    def batch(self,indices):
        episode=torch.bucketize(indices,self.ends,right=True)
        windows=torch.maximum(indices[:,None]-self.offsets[None,:],self.starts[episode,None])
        return self.x[windows],self.y[indices],self.low[indices]


def recovery_eligible_indices(lengths,original_count,span):
    # 恢复导出没有接管前历史：前150ms仅作上下文，不在这些端点上求训练损失。
    result=[];offset=original_count
    for n in lengths:
        if n>span:result.append(np.arange(offset+span,offset+n,dtype=np.int64))
        offset+=n
    if not result:raise ValueError('Recovery clips shorter than history span')
    return torch.from_numpy(np.concatenate(result))


def validation_indices(lengths,stride=10):
    # 每条轨迹固定取相同端点，首尾保留；不跨目标，所有验证目标都保留。
    result=[];offset=0
    for n in lengths:
        local=np.unique(np.r_[np.arange(0,n,stride),n-1]).astype(np.int64)
        result.append(local+offset);offset+=n
    return np.concatenate(result)
