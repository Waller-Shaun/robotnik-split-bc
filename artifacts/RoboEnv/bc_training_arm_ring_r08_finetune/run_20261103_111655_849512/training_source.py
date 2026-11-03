"""[Arm finetune change] 只微调机械臂，绝不训练底盘或运行闭环测试。

继承原26->256->256->128->11模型及原归一化；损失仅使用后7维关节力矩。
低高度z<=0.35m与其他高度各占期望50%采样，原始NPZ、划分及底盘模型不改。
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from train_bc_26d_noise_ring_bodyframe import BCPolicy  # 只复用类，旧main不会运行。
from robotnik_bodyframe_features import BODYFRAME_FEATURE_VERSION, build_bodyframe_observations

SEED = 42
BATCH_SIZE = 4096
MAX_EPOCHS = 30
LEARNING_RATE = 1e-5
PATIENCE = 8
LOW_MAX_Z = .35
LOW_SAMPLE_FRACTION = .5
SAMPLES_PER_EPOCH = 500_000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_files(manifest, split, limit):
    pairs=list(zip(manifest["splits"][split],manifest["targets"][split]))
    if limit is None:
        return pairs
    # 程序小批检查也保留两类高度；正式训练不删文件、不下采样原始帧。
    low=[p for p in pairs if p[1][2]<=LOW_MAX_Z+1e-6]
    other=[p for p in pairs if p[1][2]>LOW_MAX_Z+1e-6]
    selected=low[:max(1,limit//2)]+other[:max(1,limit-limit//2)]
    if len(selected)!=limit:
        raise ValueError("Insufficient files for stratified smoke selection")
    return selected


def load_dataset(manifest, split, stats, max_files=None):
    xs,ys,groups=[],[],[]
    files=select_files(manifest,split,max_files)
    for index,(filename,target) in enumerate(files,1):
        with np.load(Path(manifest["data_dir"])/filename,allow_pickle=False) as d:
            obs=d["observations"].astype(np.float32); actions=d["actions"]
            if actions.shape!=(len(obs),11) or not len(obs) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid action labels: {filename}")
            if not np.isin(obs[:,22],[0,1]).all():
                raise ValueError(f"Invalid source phase: {filename}")
            np.testing.assert_allclose(d["target"],target,rtol=0,atol=1e-6)
            x=build_bodyframe_observations(obs,d["ee_positions"])
            # [Arm finetune change] 使用继承的统计量，不重新计算或偷偷换坐标定义。
            x-=stats["x_mean"]; x/=stats["x_std"]
            y=(actions[:,4:11].astype(np.float32)-stats["y_mean"][4:11])/stats["y_std"][4:11]
            xs.append(x);ys.append(y)
            # 使用转换之前的世界目标高度；不使用新坐标系中的相对高度。
            groups.append(obs[:,21]<=LOW_MAX_Z+1e-6)
        if index%100==0 or index==len(files):
            print(f"{split}: {index}/{len(files)} trajectories loaded",flush=True)
    x,y,low=np.concatenate(xs),np.concatenate(ys),np.concatenate(groups)
    if not low.any() or low.all():
        raise ValueError("Both low and other height groups are required")
    return x,y,low


def make_loader(x,y,low,use_cuda,training):
    data=TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(low))
    sampler=None
    if training:
        weights=np.where(low,LOW_SAMPLE_FRACTION/low.sum(),(1-LOW_SAMPLE_FRACTION)/(~low).sum())
        sampler=WeightedRandomSampler(torch.from_numpy(weights),min(SAMPLES_PER_EPOCH,len(x)),replacement=True,
                                      generator=torch.Generator().manual_seed(SEED))
    return DataLoader(data,batch_size=BATCH_SIZE,sampler=sampler,shuffle=False,num_workers=0,pin_memory=use_cuda)


def arm_only_loss(prediction,labels):
    """[Arm finetune change] 前4维输出不参与损失，底盘网络也不在此计算图中。"""
    return (prediction[:,4:11]-labels).square().mean()


def run_epoch(model,loader,device,optimizer=None):
    model.train(optimizer is not None)
    sums=np.zeros(2);counts=np.zeros(2,dtype=np.int64)
    with torch.set_grad_enabled(optimizer is not None):
        for x,y,low in loader:
            x,y,low=(a.to(device,non_blocking=True) for a in (x,y,low))
            prediction=model(x)
            error=(prediction[:,4:11]-y).square()
            loss=error.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite arm loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            for k,mask in enumerate([low,~low]):
                sums[k]+=error[mask].detach().sum().item();counts[k]+=int(mask.sum().item())
    if np.any(counts==0):
        raise RuntimeError("Epoch must contain both height groups")
    per_group=sums/(counts*7)
    return {"arm_mse":float(sums.sum()/(counts.sum()*7)),"low_arm_mse":float(per_group[0]),
            "other_arm_mse":float(per_group[1]),"balanced_arm_mse":float(per_group.mean())}


def main():
    roboenv=Path(__file__).resolve().parents[3]/"RoboEnv"
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nav-run-dir",type=Path,default=roboenv/"bc_training_nav_ring_r08/run_20261103_022339_570160")
    parser.add_argument("--epochs",type=int,default=MAX_EPOCHS)
    parser.add_argument("--output-dir",type=Path,default=None)
    parser.add_argument("--max-train-files",type=int,default=None,help="Program smoke check only")
    parser.add_argument("--max-val-files",type=int,default=None,help="Program smoke check only")
    args=parser.parse_args()
    if args.epochs<=0 or any(v is not None and v<2 for v in [args.max_train_files,args.max_val_files]):
        parser.error("epochs must be positive and file limits at least 2")
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    cuda=torch.cuda.is_available();device=torch.device("cuda" if cuda else "cpu")
    if cuda:torch.cuda.manual_seed_all(SEED)
    else:torch.set_num_threads(min(4,torch.get_num_threads()))
    nav=args.nav_run_dir.resolve()
    nav_config=json.loads((nav/"config.json").read_text(encoding="utf-8"))
    parent=Path(nav_config["arm_run_dir"])
    manifest=json.loads((nav/"data_split.json").read_text(encoding="utf-8"))
    if {k:len(v) for k,v in manifest["targets"].items()}!={"train":558,"val":116,"test":124}:
        raise ValueError("Expected unchanged 798-target r>=0.8m split")
    parent_checkpoint=torch.load(parent/"best_model.pt",map_location="cpu",weights_only=True)
    parent_config=json.loads((parent/"config.json").read_text(encoding="utf-8"))
    if parent_checkpoint["config"]!=parent_config or parent_config.get("input_feature_version")!=BODYFRAME_FEATURE_VERSION:
        raise ValueError("Parent arm checkpoint/schema differs")
    for name,expected in nav_config["arm_file_sha256"].items():
        if sha256(parent/name)!=expected:
            raise ValueError(f"Parent arm file changed: {name}")
    with np.load(parent/"normalization.npz",allow_pickle=False) as d:
        stats={k:d[k].copy() for k in ["x_mean","x_std","y_mean","y_std"]}
    if (stats["x_mean"].shape!=(26,) or stats["x_std"].shape!=(26,) or stats["y_mean"].shape!=(11,)
            or stats["y_std"].shape!=(11,) or not all(np.isfinite(a).all() for a in stats.values())
            or np.any(stats["x_std"]<=0) or np.any(stats["y_std"]<=0)
            or stats["x_mean"][22]!=0 or stats["x_std"][22]!=1):
        raise ValueError("Invalid parent normalization")
    frozen_nav={n:sha256(nav/n) for n in ["best_model.pt","normalization.npz","config.json","data_split.json"]}
    print(f"Device: {device}; arm-only finetune; navigation weights remain unchanged",flush=True)
    x,y,low=load_dataset(manifest,"train",stats,args.max_train_files)
    xv,yv,lowv=load_dataset(manifest,"val",stats,args.max_val_files)
    tr_loader=make_loader(x,y,low,cuda,True);va_loader=make_loader(xv,yv,lowv,cuda,False)
    run=(args.output_dir or roboenv/"bc_training_arm_ring_r08_finetune"/datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")).resolve()
    run.mkdir(parents=True,exist_ok=False)
    # 保留推理所需原特征配置，移除与本次无关的旧整机训练／重点采样记录。
    config={k:v for k,v in parent_config.items() if k not in ["sampling_summary","sampling_method","focus_phase","focus_sample_weight",
             "low_target_max_z_m","near_target_max_error_m","wheel_weight","stop_weight","train_trajectories","val_trajectories",
             "train_samples","val_samples","samples_per_epoch"]}
    config.update({"training_mode":"arm_only_finetune_navigation_frozen","experiment_variant":"arm_low_z035_replay_r08",
                   "pretrained_weights_loaded":True,"source_model_run_dir":str(parent),"source_model_epoch":parent_checkpoint["epoch"],
                   "nav_run_dir":str(nav),"frozen_nav_file_sha256":frozen_nav,"navigation_weights_updated":False,
                   "source_split_path":str(nav/"data_split.json"),"source_split_sha256":sha256(nav/"data_split.json"),
                   "split_source_dir":str(nav),"target_counts":{k:len(v) for k,v in manifest["targets"].items()},
                   "minimum_horizontal_radius_m":.8,"train_trajectories":len(select_files(manifest,"train",args.max_train_files)),
                   "val_trajectories":len(select_files(manifest,"val",args.max_val_files)),"train_samples":len(x),"val_samples":len(xv),
                   "loss_type":"mse_of_normalized_arm_outputs_only","trained_output_columns":list(range(4,11)),"trained_output_dim":7,
                   "wheel_outputs_used":False,"selection_metric":"val_balanced_arm_mse",
                   "validation_selection_definition":"0.5*low_height_arm_mse + 0.5*other_height_arm_mse",
                   "low_height_max_z_m":LOW_MAX_Z,"expected_low_training_fraction":LOW_SAMPLE_FRACTION,
                   "natural_low_training_fraction":float(low.mean()),"natural_low_validation_fraction":float(lowv.mean()),
                   "sampling_method":"WeightedRandomSampler_low_other_50_50","sampling_replacement":True,
                   "samples_per_epoch":min(SAMPLES_PER_EPOCH,len(x)),"sampling_seed":SEED,
                   "normalization_distribution":"copied_parent_arm_statistics_not_recomputed",
                   "source_normalization_sha256":sha256(parent/"normalization.npz"),
                   "batch_size":BATCH_SIZE,"max_epochs":args.epochs,"learning_rate":LEARNING_RATE,"patience":PATIENCE,
                   "seed":SEED,"device":str(device),"torch_version":str(torch.__version__),
                   "training_metrics_distribution":"low_other_height_balanced_sampled_frames",
                   "validation_metrics_distribution":"all_original_r08_validation_frames_no_resampling",
                   "smoke_only":args.max_train_files is not None or args.max_val_files is not None})
    (run/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    (run/"data_split.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    shutil.copyfile(parent/"normalization.npz",run/"normalization.npz")
    (run/"training_source.py").write_text(Path(__file__).read_text(encoding="utf-8"),encoding="utf-8")
    model=BCPolicy().to(device);model.load_state_dict(parent_checkpoint["model_state_dict"],strict=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    initial=run_epoch(model,va_loader,device)
    best=initial["balanced_arm_mse"];best_epoch=0;stale=0

    def save_best(epoch,metrics):
        torch.save({"model_state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"epoch":epoch,
                    "val_mse":metrics["arm_mse"],"val_loss":metrics["balanced_arm_mse"],"val_metrics":metrics,
                    "config":config},run/"best_model.pt")

    save_best(0,initial)  # 若微调没有改善验证指标，可回退到继承权重。
    (run/"initial_validation.json").write_text(json.dumps(initial,indent=2),encoding="utf-8")
    metrics=["arm_mse","low_arm_mse","other_arm_mse","balanced_arm_mse"]
    print(f"Output: {run}\nInitial validation: {initial}",flush=True)
    with (run/"history.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f);writer.writerow(["epoch","seconds"]+[s+"_"+m for s in ["train","val"] for m in metrics])
        for epoch in range(1,args.epochs+1):
            for name,expected in frozen_nav.items():
                if sha256(nav/name)!=expected:
                    raise RuntimeError(f"Frozen navigation file changed: {name}")
            start=time.perf_counter()
            tr=run_epoch(model,tr_loader,device,optimizer);va=run_epoch(model,va_loader,device)
            writer.writerow([epoch,time.perf_counter()-start]+[a[m] for a in [tr,va] for m in metrics]);f.flush()
            print(f"Epoch {epoch:03d} | train arm MSE {tr['arm_mse']:.6f} | val arm MSE {va['arm_mse']:.6f} | low {va['low_arm_mse']:.6f} | other {va['other_arm_mse']:.6f} | balanced {va['balanced_arm_mse']:.6f}",flush=True)
            if va["balanced_arm_mse"]<best:
                best=va["balanced_arm_mse"];best_epoch=epoch;stale=0;save_best(epoch,va)
            else:stale+=1
            if stale>=PATIENCE:
                print("Early stopping: balanced validation arm MSE did not improve.",flush=True);break
    for name,expected in frozen_nav.items():
        if sha256(nav/name)!=expected:raise RuntimeError("Navigation changed during arm finetune")
    (run/"training_finished.json").write_text(json.dumps({"best_epoch":best_epoch,"last_epoch":epoch,
        "best_balanced_validation_arm_mse":best,"navigation_unchanged":True,"smoke_only":config["smoke_only"]},indent=2),encoding="utf-8")
    print(f"Best finetune epoch: {best_epoch}; balanced validation arm MSE: {best:.6f}\nResults: {run}",flush=True)
    print("Training finished; run batch_test_bc_split_nav_arm_finetuned.py separately.",flush=True)


if __name__=="__main__":
    main()
