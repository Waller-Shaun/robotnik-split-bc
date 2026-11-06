"""Lightweight training-target selection and sensor-stall detection."""
import numpy as np

def select_training_targets(manifest, count=20):
    """[Recovery probe] 只按训练坐标取样，覆盖最低层、0.35层和补采连续低高度。"""
    if count < 1:
        raise ValueError("cases must be positive")
    targets=manifest['targets']['train']
    groups=[[],[],[]]
    for index,t in enumerate(targets):
        if abs(t[2]-.2)<1e-6:groups[0].append((index,t))
        elif abs(t[2]-.35)<1e-6:groups[1].append((index,t))
        elif .2<t[2]<.5:groups[2].append((index,t))
    ordered=[]
    for group in groups:
        group.sort(key=lambda v:(np.arctan2(v[1][1],v[1][0]),np.hypot(v[1][0],v[1][1])))
        # 角度顺序上均匀分散后交错取点，避免仅抽取清单最前端。
        n=len(group)
        order=np.argsort(np.mod(np.arange(n)*.6180339887498949,1))
        ordered.append([group[i] for i in order])
    selected=[]
    while len(selected)<count and any(ordered):
        for group in ordered:
            if group and len(selected)<count:selected.append(group.pop(0))
    if len(selected)!=count:raise ValueError('Not enough low-height training targets')
    return [{'train_index':i,'target':t,'seed':30042+i} for i,t in selected]

def stalled_observations(history, now, near=.05, window=2., improvement=.002):
    """使用带噪估计误差触发，不把真实状态输入策略或专家。"""
    if not history or now-history[0][0]<window-1e-8:return False
    times=np.array([p[0] for p in history]);errors=np.array([p[1] for p in history])
    first=errors[times<=now-window+.5];last=errors[times>=now-.5]
    if not len(first) or not len(last):return False
    recent=float(np.median(last))
    return .01<=recent<=near and float(np.median(first))-recent<improvement
