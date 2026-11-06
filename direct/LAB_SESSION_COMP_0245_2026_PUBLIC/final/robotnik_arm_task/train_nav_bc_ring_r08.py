"""[Split policy change] 只训练新底盘网络，不运行仿真或测试，不更新机械臂。

数据：半径>=0.8m、原归属不变的558个训练目标及116个验证目标。
输入8维，输出FR/FL/BL/BR四轮速度；机械臂复用原26维模型的后7维输出。
为节省时间，每10步保留一帧，并额外保留所有阶段切换前后帧；四阶段平衡采样。
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from robotnik_navigation_policy import NavigationPolicy, NAV_FEATURE_VERSION, NAV_PARAMS, PHASE_NAMES, reconstruct_navigation
from robotnik_bodyframe_features import BODYFRAME_FEATURE_VERSION

SEED = 42
BATCH_SIZE = 8192
MAX_EPOCHS = 120
LEARNING_RATE = 1e-3
PATIENCE = 12
STOP_WEIGHT = 2.0
FRAME_STRIDE = 10
SAMPLES_PER_EPOCH = 250_000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(manifest, split, stride=FRAME_STRIDE, max_files=None):
    """读取原NPZ，验证阶段，再在内存派生底盘训练样本；不改源文件。"""
    xs, ys, phases = [], [], []
    report = {"trajectories":0,"raw_steps":0,"sampled_steps":0,"phase_mismatch_steps":0,
              "max_rebuilt_wheel_error_rad_s":0.,"raw_phase_counts":[0]*4}
    files = manifest["splits"][split][:max_files]
    for index, filename in enumerate(files, 1):
        try:
            with np.load(Path(manifest["data_dir"])/filename, allow_pickle=False) as data:
                obs, actions = data["observations"], data["actions"]
                x, phase, audit = reconstruct_navigation(obs, actions, data["timestamps"], data["target"], data["base_orientations"])
                changes = np.flatnonzero(np.diff(phase)) + 1
                indices = np.unique(np.clip(np.r_[np.arange(0,len(obs),stride), changes-1, changes, changes+1, 0, len(obs)-1],0,len(obs)-1)).astype(int)
                xs.append(x[indices]); ys.append(actions[indices,:4].astype(np.float32)); phases.append(phase[indices])
                report["trajectories"] += 1; report["raw_steps"] += len(obs); report["sampled_steps"] += len(indices)
                report["raw_phase_counts"] = (np.array(report["raw_phase_counts"])+np.bincount(phase,minlength=4)).tolist()
                report["max_rebuilt_wheel_error_rad_s"] = max(report["max_rebuilt_wheel_error_rad_s"],audit["max_rebuilt_wheel_error_rad_s"])
        except Exception as exc:
            raise ValueError(f"{filename}: {exc}") from exc
        if index % 100 == 0 or index == len(files):
            print(f"{split}: {index}/{len(files)} trajectories, phase/label checks passed",flush=True)
    if not xs:
        raise ValueError("Empty dataset")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(phases), report


def compute_stats(array):
    mean = array.mean(0,dtype=np.float64).astype(np.float32)
    std = array.std(0,dtype=np.float64).astype(np.float32)
    std[std<1e-6] = 1.
    return mean,std


def make_loader(x,y,phase,use_cuda,training):
    dataset=TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(phase.astype(np.int64)))
    sampler=None
    if training:
        counts=np.bincount(phase,minlength=4)
        weights=1.0/counts[phase]
        stop_fraction=float(weights[phase==3].sum()/weights.sum())
        sampler=WeightedRandomSampler(torch.from_numpy(weights),min(SAMPLES_PER_EPOCH,len(x)),replacement=True,
                                      generator=torch.Generator().manual_seed(SEED))
    else:
        stop_fraction=float((phase==3).mean())
    loader=DataLoader(dataset,batch_size=BATCH_SIZE,sampler=sampler,shuffle=False,num_workers=0,pin_memory=use_cuda)
    loader.stop_fraction=stop_fraction
    return loader


def run_epoch(model,loader,device,optimizer=None):
    training=optimizer is not None
    model.train(training)
    sums=np.zeros(4,dtype=np.float64); counts=np.zeros(4,dtype=np.int64)
    with torch.set_grad_enabled(training):
        for x,y,phase in loader:
            x,y,phase=(v.to(device,non_blocking=True) for v in (x,y,phase))
            error=(model(x)-y).square()
            stop=error[phase==3].sum()/(len(x)*4*loader.stop_fraction) if loader.stop_fraction>0 else error.sum()*0
            loss=error.mean()+STOP_WEIGHT*stop
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training/validation loss")
            if training:
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            for k in range(4):
                mask=phase==k
                sums[k]+=error[mask].detach().sum().item();counts[k]+=int(mask.sum().item())
    mse=float(sums.sum()/(4*counts.sum()))
    per_phase=np.divide(sums,counts*4,out=np.zeros(4),where=counts>0)
    return {"loss":mse+STOP_WEIGHT*float(per_phase[3]),"mse":mse,
            **{name+"_mse":float(per_phase[i]) for i,name in enumerate(PHASE_NAMES)}}


def main():
    roboenv=Path(__file__).resolve().parents[3]/"RoboEnv"
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-path",type=Path,default=roboenv/"bc_training_nav_ring_r08/data_split.json")
    parser.add_argument("--arm-run-dir",type=Path,default=roboenv/"bc_training_26d_noise_ring_bodyframe/run_20261102_184221_743102")
    parser.add_argument("--epochs",type=int,default=MAX_EPOCHS)
    parser.add_argument("--output-dir",type=Path,default=None)
    # 以下两个选项只用于程序小批检查；使用时明确标为smoke，不能充当正式训练。
    parser.add_argument("--max-train-files",type=int,default=None)
    parser.add_argument("--max-val-files",type=int,default=None)
    args=parser.parse_args()
    if args.epochs<=0 or any(v is not None and v<=0 for v in [args.max_train_files,args.max_val_files]):
        parser.error("epochs/file limits must be positive")
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    use_cuda=torch.cuda.is_available();device=torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:torch.cuda.manual_seed_all(SEED)
    else:torch.set_num_threads(min(torch.get_num_threads(),4))
    manifest=json.loads(args.split_path.read_text(encoding="utf-8"))
    if (manifest.get("minimum_horizontal_radius_m")!=.8
            or {k:len(v) for k,v in manifest["splits"].items()}!={"train":558,"val":116,"test":124}):
        raise ValueError("Expected the frozen 798-target r>=0.8m manifest")
    arm_dir=args.arm_run_dir.resolve()
    arm_config=json.loads((arm_dir/"config.json").read_text(encoding="utf-8"))
    if arm_config.get("input_feature_version")!=BODYFRAME_FEATURE_VERSION:
        raise ValueError("Expected the pretrained body-frame arm model")
    # 必须保证复用手臂没有见过新的验证或测试目标。
    arm_manifest=json.loads((arm_dir/"data_split.json").read_text(encoding="utf-8"))
    key=lambda t:tuple(round(float(v),6) for v in t)
    old_train={key(t) for t in arm_manifest["targets"]["train"]}
    if old_train & {key(t) for split in ["val","test"] for t in manifest["targets"][split]}:
        raise ValueError("Arm pretraining overlaps held-out target groups")
    print(f"Device: {device}; nav 8 -> 64 -> 64 -> 4; pretrained arm stays frozen",flush=True)
    print("Reconstructing phases and verifying original wheel labels...",flush=True)
    x,y,p,train_report=load_dataset(manifest,"train",max_files=args.max_train_files)
    xv,yv,pv,val_report=load_dataset(manifest,"val",max_files=args.max_val_files)
    xm,xs=compute_stats(x);ym,ys=compute_stats(y)
    xm[4:]=0;xs[4:]=1  # one-hot不标准化。
    x=(x-xm)/xs;xv=(xv-xm)/xs;y=(y-ym)/ys;yv=(yv-ym)/ys
    train_loader=make_loader(x,y,p,use_cuda,True);val_loader=make_loader(xv,yv,pv,use_cuda,False)
    stamp=datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run=(args.output_dir or roboenv/"bc_training_nav_ring_r08"/stamp).resolve()
    run.mkdir(parents=True,exist_ok=False)
    feature_file=Path(__file__).with_name("robotnik_navigation_policy.py")
    controller_file=Path(__file__).with_name("robotnik_arm_control.py")
    config={"training_mode":"navigation_from_scratch_pretrained_arm_frozen","input_dim":8,"output_dim":4,
            "input_feature_version":NAV_FEATURE_VERSION,"hidden_dims":[64,64],"activation":"ReLU",
            "feature_columns":["parking_goal_body_x","parking_goal_body_y","heading_to_parking_error_rad","final_heading_error_rad","align","drive","orient","done"],
            "navigation_parameters":NAV_PARAMS,"phase_source":"reconstructed_original_state_machine_checked_against_phase_and_actions",
            "feature_module_sha256":sha256(feature_file),"controller_sha256":sha256(controller_file),
            "source_split_path":str(args.split_path.resolve()),"source_split_sha256":sha256(args.split_path),
            "data_dir":manifest["data_dir"],"target_counts":{k:len(v) for k,v in manifest["targets"].items()},
            "minimum_horizontal_radius_m":.8,"frame_stride":FRAME_STRIDE,"transition_neighbor_frames_preserved":True,
            "validation_temporally_subsampled":True,"phase_balanced_training":True,"sampling_replacement":True,
            "samples_per_epoch":min(SAMPLES_PER_EPOCH,len(x)),"batch_size":BATCH_SIZE,"max_epochs":args.epochs,
            "learning_rate":LEARNING_RATE,"patience":PATIENCE,"stop_weight":STOP_WEIGHT,"seed":SEED,
            "scheduler":"ReduceLROnPlateau factor=0.5 patience=4 min_lr=1e-5","selection_metric":"val_loss",
            "train_report":train_report,"val_report":val_report,"sampled_train_phase_counts":np.bincount(p,minlength=4).tolist(),
            "arm_run_dir":str(arm_dir),"arm_feature_version":BODYFRAME_FEATURE_VERSION,"arm_weights_updated":False,
            "arm_pretrained_on_original_train_targets":len(old_train),"arm_pretraining_includes_excluded_near_train_targets":True,
            "arm_file_sha256":{n:sha256(arm_dir/n) for n in ["best_model.pt","normalization.npz","config.json"]},
            "smoke_only":args.max_train_files is not None or args.max_val_files is not None,"device":str(device),"torch_version":str(torch.__version__)}
    (run/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    (run/"data_split.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    (run/"navigation_feature_source.py").write_text(feature_file.read_text(encoding="utf-8"),encoding="utf-8")
    (run/"training_source.py").write_text(Path(__file__).read_text(encoding="utf-8"),encoding="utf-8")
    np.savez(run/"normalization.npz",x_mean=xm,x_std=xs,y_mean=ym,y_std=ys)
    model=NavigationPolicy().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=4,min_lr=1e-5)
    best=float("inf");best_epoch=0;stale=0
    print(f"Output: {run}",flush=True)
    print(f"Sampled steps: train={len(x):,}, val={len(xv):,}; phase-balanced draws/epoch={config['samples_per_epoch']:,}",flush=True)
    metrics=["loss","mse","align_mse","drive_mse","orient_mse","done_mse"]
    with (run/"history.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f);writer.writerow(["epoch","learning_rate","seconds"]+[s+"_"+m for s in ["train","val"] for m in metrics])
        for epoch in range(1,args.epochs+1):
            start=time.perf_counter();lr=optimizer.param_groups[0]["lr"]
            tr=run_epoch(model,train_loader,device,optimizer);va=run_epoch(model,val_loader,device)
            writer.writerow([epoch,lr,time.perf_counter()-start]+[a[m] for a in [tr,va] for m in metrics]);f.flush()
            print(f"Epoch {epoch:03d} | train loss {tr['loss']:.6f} | val loss {va['loss']:.6f} | orient MSE {va['orient_mse']:.6f} | done MSE {va['done_mse']:.6f} | lr {lr:g}",flush=True)
            if va["loss"]<best:
                best=va["loss"];best_epoch=epoch;stale=0
                torch.save({"model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                            "epoch":epoch,"val_loss":best,"val_metrics":va,"config":config},run/"best_model.pt")
            else:stale+=1
            scheduler.step(va["loss"])
            if stale>=PATIENCE:
                print("Early stopping: no validation-loss improvement.",flush=True);break
    (run/"training_finished.json").write_text(json.dumps({"best_epoch":best_epoch,"last_epoch":epoch,"val_loss":best,"smoke_only":config["smoke_only"]},indent=2),encoding="utf-8")
    print(f"Best epoch: {best_epoch}; val loss: {best:.6f}\nResults: {run}",flush=True)
    print("Training only. Run batch_test_bc_split_nav_arm.py separately for closed-loop evaluation.",flush=True)


if __name__=="__main__":
    main()
