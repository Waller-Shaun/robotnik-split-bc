"""Offline diagnosis of one BC failure against an expert episode.

[Failure diagnosis] 不运行仿真、不训练模型；自动按目标匹配已有轨迹。
蓝/绿动作曲线使用相同专家输入；橙色闭环曲线来自另一条实际状态轨迹。
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

DEFAULT_RUN = Path(r"F:\UCL\COMP0245 Labs\RoboEnv\bc_training_23d_noise_wide_extended\run_20261031_130737_274460")


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


class Policy(torch.nn.Module):
    # [Failure diagnosis] 从检查点读取隐藏层，兼容双128和256/256/128网络。
    def __init__(self, hidden_dims):
        super().__init__()
        dims = [23, *hidden_dims, 11]
        layers = []
        for i, (before, after) in enumerate(zip(dims, dims[1:])):
            layers.append(torch.nn.Linear(before, after))
            if i < len(dims)-2:
                layers.append(torch.nn.ReLU())
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def find_rollout(run, target):
    # 优先查找最近批量评估的CSV，避免误选其他目标的“最新文件”。
    csvs = sorted((run / "batch_tests").glob("*/results.csv"), key=lambda p:p.stat().st_mtime, reverse=True)
    for path in csvs:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                value = [float(row["target_"+a]) for a in "xyz"]
                if np.allclose(value, target, atol=1e-6, rtol=0) and row.get("rollout_file"):
                    result = Path(row["rollout_file"])
                    if result.is_file():
                        return result
    for path in sorted((run / "rollouts").glob("*.npz"), key=lambda p:p.stat().st_mtime, reverse=True):
        with np.load(path, allow_pickle=False) as data:
            if np.allclose(data["target"], target, atol=1e-6, rtol=0):
                return path
    raise FileNotFoundError(f"No saved rollout found for target {target.tolist()}")


def find_expert(manifest, manifest_path, target):
    folder = Path(manifest["data_dir"])
    if not folder.is_absolute():
        folder = manifest_path.parent / folder
    matches = []
    for split, names in manifest["splits"].items():
        for name in names:
            path = folder / name
            with np.load(path, allow_pickle=False) as data:
                if np.allclose(data["target"], target, atol=1e-6, rtol=0):
                    matches.append((path, split))
    if len(matches) != 1:
        raise ValueError(f"Found {len(matches)} matching experts; specify --expert if needed")
    return matches[0]


def phase_time(times, observations):
    indices = np.flatnonzero(observations[:, 22] == 0)
    return float(times[indices[0]]) if len(indices) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--target", type=float, nargs=3, default=[.2,1.1,.4])
    parser.add_argument("--rollout", type=Path, default=None)
    parser.add_argument("--expert", type=Path, default=None)
    parser.add_argument("--start", type=float, default=4.)
    parser.add_argument("--end", type=float, default=12.)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    target = np.array(args.target, dtype=float)
    if not np.isfinite([*target,args.start,args.end]).all() or args.end <= args.start:
        parser.error("Finite target and start < end required")
    run = args.run_dir.resolve()
    manifest_path = run / "data_split.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    rollout_path = args.rollout or find_rollout(run, target)
    rollout = read_npz(rollout_path)
    expert_path, expert_split = (args.expert, "explicit") if args.expert else find_expert(manifest, manifest_path, target)
    expert = read_npz(expert_path)
    for episode in (expert, rollout):
        if not np.allclose(episode["target"], target, atol=1e-6, rtol=0):
            raise ValueError("Requested target differs from saved target")
    if "model_run" in rollout and Path(str(rollout["model_run"].item())).resolve() != run:
        raise ValueError("Rollout belongs to another model run")
    ex, ey, et = expert["observations"], expert["actions"], expert["timestamps"]
    rx, ra = rollout["observations"], rollout["actions"]
    rt = rollout["timestamps"][:len(rx)]
    if ex.shape != (len(et),23) or ey.shape != (len(et),11) or rx.shape != (len(ra),23) or ra.shape[1:] != (11,):
        raise ValueError("Input/action/time arrays are not aligned")
    norm = read_npz(run/"normalization.npz")
    ck = torch.load(run/"best_model.pt", map_location="cpu", weights_only=True)
    config = ck["config"]
    if config["input_dim"] != 23 or config["output_dim"] != 11 or config.get("activation","ReLU") != "ReLU":
        raise ValueError("Expected a 23-input/11-output ReLU model")
    torch.set_num_threads(1)
    model = Policy(config["hidden_dims"]).eval()
    model.load_state_dict(ck["model_state_dict"], strict=True)
    pred = np.empty_like(ey)
    with torch.inference_mode():
        for start in range(0,len(ex),4096):
            end = start+4096
            x = ((ex[start:end]-norm["x_mean"])/norm["x_std"]).astype(np.float32)
            pred[start:end] = model(torch.from_numpy(x)).numpy()*norm["y_std"]+norm["y_mean"]
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite model predictions")
    output = args.output_dir or run/"diagnostics"/datetime.now().strftime("failure_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True,exist_ok=False)
    ep,rp = phase_time(et,ex),phase_time(rt,rx)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    def decorate(ax):
        if ep is not None:ax.axvline(ep,color="#287cc0",ls=":",alpha=.6)
        if rp is not None:ax.axvline(rp,color="#df7727",ls=":",alpha=.6)
        ax.set_xlim(args.start,args.end);ax.grid(alpha=.2);ax.set_xlabel("Simulation time (s)")

    def panels(name,title,labels,curves,ylabel):
        count=int(np.ceil(len(labels)/2))
        fig,axs=plt.subplots(count,2,figsize=(13,2.5*count),squeeze=False,constrained_layout=True)
        for i,label in enumerate(labels):
            ax=axs.flat[i]
            for t,v,legend,color,style in curves:
                shown=(t>=args.start)&(t<=args.end)
                ax.plot(t[shown],v[shown,i],label=legend,color=color,ls=style,lw=1.)
            ax.set_title(label);ax.set_ylabel(ylabel);decorate(ax)
        for ax in list(axs.flat)[len(labels):]:ax.set_visible(False)
        axs.flat[0].legend(fontsize=8)
        fig.suptitle(title+"\nDotted lines: expert / BC navigation completion",fontsize=12)
        fig.savefig(output/name,dpi=150);plt.close(fig)

    # [Failure diagnosis] 同时刻的两条闭环轨迹不是同状态误差，不能逐点当作动作监督标签。
    panels("02_joint_angles.png","Joint angles on separate closed-loop trajectories",[f"J{i+1}" for i in range(7)],
        [(et,ex[:,5:12],"Expert measured","#287cc0","-"),(rt,rx[:,5:12],"BC measured","#df7727","--")],"Angle (rad)")
    panels("03_joint_velocities.png","Joint velocities on separate closed-loop trajectories",[f"J{i+1}" for i in range(7)],
        [(et,ex[:,12:19],"Expert measured","#287cc0","-"),(rt,rx[:,12:19],"BC measured","#df7727","--")],"Speed (rad/s)")
    curves=[(et,ey,"Expert action","#287cc0","-"),(et,pred,"BC on SAME expert states (unclipped)","#209462","--"),
            (rt,ra,"BC rollout applied (different states)","#df7727","-")]
    panels("04_arm_torques.png","Arm torque: offline fitting versus actual closed-loop execution",[f"J{i+1}" for i in range(7)],
           [(t,v[:,4:],n,c,s) for t,v,n,c,s in curves],"Torque (N m)")
    panels("05_wheel_commands.png","Wheel commands: offline fitting versus actual closed-loop execution",
           ["Front right","Front left","Back left","Back right"],[(t,v[:,:4],n,c,s) for t,v,n,c,s in curves],"Wheel speed (rad/s)")

    # 误差的正负方向能帮助区分高度偏差与水平偏差。
    et_all=et
    rt_all=rollout["timestamps"]
    ee=expert["ee_errors"]*100
    re=rollout["ee_errors"]*100
    shown_expert=(et_all>=args.start)&(et_all<=args.end)
    shown_rollout=(rt_all>=args.start)&(rt_all<=args.end)
    fig,axs=plt.subplots(4,1,figsize=(12,11),constrained_layout=True)
    axs[0].plot(et_all[shown_expert],ee[shown_expert],label="Expert EE error",color="#287cc0")
    axs[0].plot(rt_all[shown_rollout],re[shown_rollout],label="BC EE error",color="#df7727")
    axs[0].axhline(1,color="#cc3344",ls="--",label="1 cm threshold")
    axs[0].set_ylabel("Distance (cm)");axs[0].legend()
    errors=target[None,:]-rollout["ee_positions"]
    for i,label in enumerate("XYZ"):
        axs[1].plot(rt_all[shown_rollout],errors[shown_rollout,i]*100,label=f"Target - EE {label}")
    axs[1].axhline(0,color="grey",lw=.7);axs[1].set_ylabel("Error component (cm)");axs[1].legend(ncol=3)
    for i,label in enumerate("xy"):
        axs[2].plot(et,ex[:,i],label=f"Expert base {label}")
        axs[2].plot(rt,rx[:,i],ls="--",label=f"BC base {label}")
    axs[2].set_ylabel("Base position (m)");axs[2].legend(ncol=2)
    axs[3].step(et,ex[:,22],where="post",label="Expert phase")
    axs[3].step(rt,rx[:,22],where="post",ls="--",label="BC phase")
    axs[3].set_ylabel("base_active");axs[3].legend()
    for ax in axs:decorate(ax)
    fig.suptitle(f"Target {target.tolist()}: error, base and phase",fontsize=13)
    fig.savefig(output/"01_error_base_phase.png",dpi=160);plt.close(fig)

    # [Failure diagnosis] 单独放大最后5cm，避免初始一米级误差掩盖1cm附近的行为。
    entered=np.flatnonzero(rollout["ee_errors"]<=.05)
    first_near_time=float(rt_all[entered[0]]) if len(entered) else float(rt_all[0])
    zoom_start=max(float(rt_all[0]),first_near_time-.5)
    fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True,constrained_layout=True)
    axs[0].plot(et,ee,label="Expert",color="#287cc0")
    axs[0].plot(rt_all,re,label="BC rollout",color="#df7727")
    axs[0].axhline(1,color="#cc3344",ls="--",label="1 cm")
    axs[0].set(ylim=(0,5),ylabel="EE distance (cm)",title="Near-target behavior through episode end (0-5 cm zoom)")
    axs[0].legend()
    for i,label in enumerate("XYZ"):
        axs[1].plot(rt_all,errors[:,i]*100,label=f"Target - EE {label}")
    axs[1].axhline(0,color="grey",lw=.8)
    axs[1].set(ylim=(-5,5),ylabel="Error component (cm)",xlabel="Simulation time (s)")
    axs[1].legend()
    for ax in axs:ax.set_xlim(zoom_start,float(rt_all[-1]));ax.grid(alpha=.2)
    fig.savefig(output/"06_near_target_zoom.png",dpi=160);plt.close(fig)

    # [Failure diagnosis] 只有pred与ey来自相同输入，才能计算真正的离线拟合误差。
    masks={"all":np.ones(len(ex),bool),"navigation":ex[:,22]==1,"arm_reach":ex[:,22]==0,
           "expert_near_5cm":(ex[:,22]==0)&(expert["ee_errors"]<=.05),
           "expert_in_1cm":(ex[:,22]==0)&(expert["ee_errors"]<.01)}
    metrics={}
    with (output/"offline_action_errors.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.writer(stream);writer.writerow(["group","samples","action","unit","rmse","mean_prediction_minus_expert"])
        for group,mask in masks.items():
            if not mask.any():continue
            delta=(pred[mask]-ey[mask]).astype(np.float64)
            rmse=np.sqrt(np.mean(delta**2,axis=0));bias=delta.mean(axis=0)
            metrics[group]={"samples":int(mask.sum()),"rmse":rmse.tolist(),"bias":bias.tolist()}
            for i in range(11):writer.writerow([group,int(mask.sum()),f"wheel_{i+1}" if i<4 else f"arm_J{i-3}","rad/s" if i<4 else "N m",rmse[i],bias[i]])
    near_index=int(np.argmin(rollout["ee_errors"]))
    near_time=float(rt_all[near_index])
    nearest_obs_index=min(near_index,len(rx)-1)
    snapshots=[]
    first_near_obs_index=min(int(np.argmin(np.abs(rt-first_near_time))),len(rx)-1)
    for when in sorted(set([first_near_time,first_near_time+1,first_near_time+2,near_time,args.end,float(rt_all[-1])])):
        if when>float(rt_all[-1]):continue
        i=int(np.argmin(np.abs(rt-when)));ei=int(np.argmin(np.abs(rt_all-when)))
        dyaw=np.arctan2(rx[i,3],rx[i,4])-np.arctan2(rx[nearest_obs_index,3],rx[nearest_obs_index,4])
        snapshots.append({"time_s":float(rt[i]),"ee_error_cm":float(re[ei]),
            "base_xy_change_from_nearest_mm":float(np.linalg.norm(rx[i,:2]-rx[nearest_obs_index,:2])*1000),
            "base_xy_change_from_first_5cm_mm":float(np.linalg.norm(rx[i,:2]-rx[first_near_obs_index,:2])*1000),
            "yaw_change_from_nearest_deg":float(np.degrees((dyaw+np.pi)%(2*np.pi)-np.pi)),
            "max_abs_wheel_command":float(np.max(np.abs(ra[i,:4]))),
            "arm_angles":rx[i,5:12].tolist()})
    summary={"run_dir":str(run),"rollout":str(rollout_path),"expert":str(expert_path),"expert_split":expert_split,
             "target":target.tolist(),"checkpoint_epoch":int(ck["epoch"]),"hidden_dims":config["hidden_dims"],
             "expert_phase_time_s":ep,"bc_phase_time_s":rp,"plot_window_s":[args.start,args.end],
             "bc_min_error_m":float(rollout["ee_errors"][near_index]),"bc_min_error_time_s":near_time,
             "bc_first_5cm_time_s":first_near_time,
             "bc_final_error_m":float(rollout["ee_errors"][-1]),"offline_metrics":metrics,"snapshots":snapshots,
             "limitations":["Expert episode has collection measurement noise; BC rollout may be noiseless.",
                            "Same-time rollout and expert curves are not matched states.",
                            "No oracle expert actions are available at BC-only states; offline fit alone does not prove the cause of instability."]}
    (output/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    lines=["# 单目标失败诊断","",f"目标：`{target.tolist()}`；模型轮次：{ck['epoch']}；专家轨迹所属集合：{expert_split}。","",
           f"模型最低误差：{summary['bc_min_error_m']*100:.4f}cm，时刻{near_time:.3f}s；最终误差：{summary['bc_final_error_m']*100:.4f}cm。","",
           "## 看图方法","",
           "1. `01_error_base_phase.png`：观察误差从哪个方向增加，以及底盘、阶段是否同时变化。",
           "2. `02_joint_angles.png`与`03_joint_velocities.png`：定位最先变化的关节。",
           "3. `04_arm_torques.png`：蓝线与绿线是同一专家输入下的标签和预测，能直接判断离线拟合；橙线是模型执行时的动作，状态已经不同。",
           "4. `05_wheel_commands.png`：查看模型在机械臂阶段是否仍输出非零轮速。","",
           "5. `06_near_target_zoom.png`：放大0～5cm并显示到轨迹结束，直接观察1cm附近是在漂移、震荡还是缓慢收敛。","",
           "`offline_action_errors.csv`给出全程、导航、机械臂、专家进入5cm和1cm范围时的逐动作RMSE及有符号偏差。误差单位为真实物理单位，不能直接把轮速和力矩大小横向比较。","",
           "如果蓝绿在专家末段就存在持续偏差，说明已有专家状态上的拟合仍有不足；若拟合较好而实际闭环偏离明显，则应进一步调查偏离状态下的控制。两者可同时存在，不能仅凭叠加曲线宣称已找到唯一原因。","",
           "专家采集含测量噪声，闭环测试可能关闭噪声；这些差异也会影响轨迹。该脚本不在模型状态上重新调用专家，不将最近专家样本冒充真实纠正标签。","",
           "![末段误差放大](06_near_target_zoom.png)","","![误差与底盘](01_error_base_phase.png)","","![关节角](02_joint_angles.png)","",
           "![关节角速度](03_joint_velocities.png)","","![机械臂力矩](04_arm_torques.png)","","![轮速](05_wheel_commands.png)"]
    (output/"诊断说明.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    np.savez_compressed(output/"offline_predictions.npz",timestamps=et,expert_actions=ey,model_on_expert_states=pred,target=target)
    print("Target:",target.tolist())
    print("Expert:",expert_path)
    print("Rollout:",rollout_path)
    print(f"Minimum error: {summary['bc_min_error_m']*100:.4f} cm at {near_time:.3f} s")
    print("Results:",output)


if __name__=="__main__":
    main()
