"""Visualize held-out target groups and the robot's initial collision surface.

No rollout or training is performed. HTML is self-contained; PNG/PDF are also saved.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

DEFAULT_ROBOENV = Path(r"F:\UCL\COMP0245 Labs\RoboEnv")
DEFAULT_RUN = DEFAULT_ROBOENV / "bc_training_23d_noise_finetune" / "run_20261030_100242_765759"
COLORS = {"train": "#2674bf", "val": "#df8b16", "test": "#b83778"}
LABELS = {"train": "Training", "val": "Validation", "test": "Test"}


def load_targets(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    groups = {}
    for split in ("train", "val", "test"):
        targets = np.asarray(manifest["targets"][split], dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 3 or not np.isfinite(targets).all():
            raise ValueError(f"Invalid targets in {split}")
        groups[split] = targets
    sets = {k: set(map(tuple, np.round(v, 8))) for k, v in groups.items()}
    if any(len(sets[k]) != len(groups[k]) for k in sets):
        raise ValueError("Duplicate targets within a split")
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("Target groups overlap")
    return groups


def robot_cloud(roboenv_dir):
    # [Visualization] Read the real robot configuration; never edit it or step physics.
    # Rays sample collision geometry, rather than pretending this is sensor data.
    from simulation_and_control import pb
    sim = pb.SimInterface("robotnik_arm.json", conf_file_path_ext=str(roboenv_dir), use_gui=False)
    client = sim.pybullet_client
    try:
        rid = sim.bot[0].bot_pybullet
        aabbs = [client.getAABB(rid, link) for link in range(-1, client.getNumJoints(rid))]
        lower = np.min([a[0] for a in aabbs], axis=0)
        upper = np.max([a[1] for a in aabbs], axis=0)
        hits = []
        for axis in range(3):
            other = [j for j in range(3) if j != axis]
            grid = np.meshgrid(*[np.linspace(lower[j] - .002, upper[j] + .002, 105) for j in other])
            for direction in (-1, 1):
                start = np.zeros((grid[0].size, 3))
                for j, coords in zip(other, grid):
                    start[:, j] = coords.ravel()
                start[:, axis] = lower[axis] - .15 if direction == 1 else upper[axis] + .15
                end = start.copy()
                end[:, axis] = upper[axis] + .15 if direction == 1 else lower[axis] - .15
                result = client.rayTestBatch(start.tolist(), end.tolist())
                hits.extend(hit[3] for hit in result if hit[0] == rid)
        cloud = np.unique(np.round(np.asarray(hits), 4), axis=0)
        if cloud.ndim != 2 or len(cloud) < 100:
            raise RuntimeError("Robot point cloud sampling failed")
        if len(cloud) > 6500:
            cloud = cloud[np.random.default_rng(42).choice(len(cloud), 6500, replace=False)]
        base_position, base_orientation = client.getBasePositionAndOrientation(rid)
        heading = client.getEulerFromQuaternion(base_orientation)[2]
        return cloud, np.asarray(base_position), heading
    finally:
        client.disconnect()


def static_plot(groups, cloud, base, heading, output, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*cloud.T, s=.7, c="#596575", alpha=.55, depthshade=False,
               label="Robot initial pose (collision surface)", rasterized=True)
    for split, points in groups.items():
        ax.scatter(*points.T, s=22, c=COLORS[split], alpha=.92, depthshade=False,
                   edgecolors="white", linewidths=.3, label=f"{LABELS[split]} ({len(points)})")
    ax.scatter(*base, color="#172b4d", marker="x", s=60, linewidth=2)
    ax.quiver(base[0], base[1], .03, .32*np.cos(heading), .32*np.sin(heading), 0,
              color="#172b4d", linewidth=2, arrow_length_ratio=.25)
    ax.text(base[0]-.1, base[1]-.22, .03, "Initial base / heading", color="#172b4d", fontsize=9)
    points = np.vstack([cloud, *groups.values()])
    lo, hi = points.min(0)-.12, points.max(0)+.12
    lo[2] = min(0, lo[2])
    ax.set(xlim=(lo[0],hi[0]), ylim=(lo[1],hi[1]), zlim=(lo[2],hi[2]),
           xlabel="World X (m)", ylabel="World Y (m)", zlabel="World Z (m)")
    ax.set_box_aspect(hi-lo)
    ax.view_init(elev=24, azim=-54)
    ax.set_title("Target distribution and robot initial pose", fontsize=19, pad=18)
    ax.legend(loc="upper left", bbox_to_anchor=(-.04,1.01), fontsize=9, framealpha=.96)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
    fig.text(.5,.035,"468 unique targets  |  0.1 m grid  |  Fixed initial pose\n"
             "Robot points represent collision geometry, not a measured depth scan.",
             ha="center", color="#566074", fontsize=10)
    fig.subplots_adjust(left=.02,right=.99,bottom=.09,top=.91)
    fig.savefig(output / "target_splits.png", dpi=220)
    fig.savefig(output / "target_splits.pdf")
    if show:
        plt.show()
    else:
        plt.close(fig)


def interactive_html(groups, cloud, base, heading, output):
    data = {k: np.round(v,5).tolist() for k,v in groups.items()}
    data.update(robot=np.round(cloud,4).tolist(), base=base.tolist(), heading=float(heading))
    html = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Targets & robot | Dataset split</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f7fb;color:#172b4d;font:15px system-ui,"Microsoft YaHei",sans-serif}
header{padding:24px 30px 12px}h1{font-size:25px;margin:0 0 8px}p{margin:6px 0;color:#62718a}
nav{padding:8px 30px;display:flex;gap:20px;align-items:center;flex-wrap:wrap}
label{cursor:pointer}button{padding:7px 14px;background:white;border:1px solid #d8dfeb;border-radius:8px;cursor:pointer}
canvas{display:block;width:100%;height:calc(100vh - 215px);min-height:440px;background:#fff;touch-action:none}
#tip{position:fixed;display:none;background:#172b4d;color:white;border-radius:6px;padding:8px;pointer-events:none;font-size:13px}
footer{padding:10px 30px;color:#62718a;font-size:12px}
</style><header><h1>Target 点云与机器人初始姿态</h1>
<p>拖动旋转 · 滚轮缩放 · 悬停查看目标坐标 · 勾选筛选数据集</p></header>
<nav><label style="color:#2674bf"><input type="checkbox" id="train" checked> 训练集 Train (327)</label>
<label style="color:#df8b16"><input type="checkbox" id="val" checked> 验证集 Validation (70)</label>
<label style="color:#b83778"><input type="checkbox" id="test" checked> 测试集 Test (71)</label>
<label><input type="checkbox" id="robot" checked> 机器人初始姿态</label>
<button id="reset">重置视角</button><button id="top">俯视图</button></nav>
<canvas id="view"></canvas><div id="tip"></div>
<footer>坐标单位：米。灰色机器人点云采自仿真碰撞表面，不是真实深度相机数据。所有点使用相同坐标尺度。</footer>
<script>
const D=__DATA__,cv=document.getElementById('view'),ctx=cv.getContext('2d'),tip=document.getElementById('tip');
const colors={train:'#2674bf',val:'#df8b16',test:'#b83778',robot:'#667587'};
let yaw=-.72,pitch=.43,zoom=1,drag=null,moved=false,screen=[];
const center=[0,.59,.39];
function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];
let u=Math.cos(yaw)*x-Math.sin(yaw)*y,v=Math.sin(yaw)*x+Math.cos(yaw)*y;
let w=Math.cos(pitch)*z-Math.sin(pitch)*v,d=Math.cos(pitch)*v+Math.sin(pitch)*z;
let s=Math.min(cv.clientWidth/2.45,cv.clientHeight/1.9)*zoom;
return [cv.clientWidth/2+u*s,cv.clientHeight/2-w*s,d]}
function line(a,b,color,width=1){a=project(a);b=project(b);ctx.strokeStyle=color;ctx.lineWidth=width;
ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke()}
function draw(){const ratio=window.devicePixelRatio||1;cv.width=cv.clientWidth*ratio;cv.height=cv.clientHeight*ratio;
ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,cv.clientWidth,cv.clientHeight);
for(let i=-4;i<=8;i++){let x=i/5;line([x,-.6,0],[x,1.6,0],'#e9edf3');}
for(let i=-3;i<=8;i++){let y=i/5;line([-.8,y,0],[.9,y,0],'#e9edf3');}
for(const [end,color,name] of [[[.45,0,0],'#c65f61','X'],[[0,.45,0],'#438b77','Y'],[[0,0,.45],'#657ac9','Z']]){
line([0,0,0],end,color,2);let q=project(end);ctx.fillStyle=color;ctx.font='13px system-ui';ctx.fillText(name,q[0]+5,q[1]-5)}
let all=[];for(const key of ['robot','train','val','test'])if(document.getElementById(key).checked)
D[key].forEach(p=>all.push({key,p,q:project(p)}));all.sort((a,b)=>a.q[2]-b.q[2]);screen=[];
for(const o of all){let r=o.key==='robot'?1:4.4;ctx.globalAlpha=o.key==='robot'?.58:.95;ctx.fillStyle=colors[o.key];
ctx.beginPath();ctx.arc(o.q[0],o.q[1],r,0,Math.PI*2);ctx.fill();
if(o.key!=='robot'){ctx.strokeStyle='#fff';ctx.lineWidth=.6;ctx.stroke();screen.push(o)}}ctx.globalAlpha=1;
if(document.getElementById('robot').checked){let a=[D.base[0],D.base[1],.02],b=[a[0]+.32*Math.cos(D.heading),a[1]+.32*Math.sin(D.heading),.02];
line(a,b,'#172b4d',3);let q=project([D.base[0]-.2,D.base[1]-.35,.02]);ctx.fillStyle='#172b4d';ctx.fillText('Initial robot pose',q[0],q[1]);}
ctx.fillStyle='#62718a';ctx.fillText('World coordinates (m)  |  grid spacing: 0.2 m',18,cv.clientHeight-18)}
cv.onpointerdown=e=>{drag=[e.clientX,e.clientY,yaw,pitch];moved=false;cv.setPointerCapture(e.pointerId);tip.style.display='none'};
cv.onpointermove=e=>{if(drag){let dx=e.clientX-drag[0],dy=e.clientY-drag[1];moved=true;yaw=drag[2]+dx*.008;pitch=Math.max(-1.55,Math.min(1.55,drag[3]+dy*.008));draw();return;}
let rect=cv.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;
let near=screen.map(o=>({o,d:Math.hypot(o.q[0]-x,o.q[1]-y)})).sort((a,b)=>a.d-b.d)[0];
if(near&&near.d<10){tip.textContent=near.o.key+'  ['+near.o.p.map(v=>v.toFixed(1)).join(', ')+'] m';tip.style.left=Math.min(e.clientX+12,innerWidth-235)+'px';tip.style.top=(e.clientY+12)+'px';tip.style.display='block'}else tip.style.display='none'};
cv.onpointerup=cv.onpointercancel=()=>drag=null;cv.onpointerleave=()=>tip.style.display='none';
cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.4,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});
for(const key of ['train','val','test','robot'])document.getElementById(key).onchange=draw;
document.getElementById('reset').onclick=()=>{yaw=-.72;pitch=.43;zoom=1;draw()};
document.getElementById('top').onclick=()=>{yaw=0;pitch=Math.PI/2;zoom=1;draw()};window.onresize=draw;draw();
</script></html>'''
    (output / "target_splits.html").write_text(html.replace("__DATA__",json.dumps(data,separators=(",",":"))),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=DEFAULT_RUN / "data_split.json")
    parser.add_argument("--roboenv-dir",type=Path,default=DEFAULT_ROBOENV)
    parser.add_argument("--output-dir",type=Path,default=None)
    parser.add_argument("--no-show",action="store_true",help="Save files without opening Matplotlib")
    args=parser.parse_args()
    groups=load_targets(args.manifest)
    cloud,base,heading=robot_cloud(args.roboenv_dir)
    output=args.output_dir or args.roboenv_dir / "target_visualizations" / datetime.now().strftime("splits_%Y%m%d_%H%M%S")
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output / "target_points.npz",**groups,robot_points=cloud,base_position=base,base_heading=heading)
    metadata={"manifest":str(args.manifest.resolve()),"counts":{k:len(v) for k,v in groups.items()},
              "robot_points":len(cloud),"robot_geometry":"collision surface, ray sampled, initial configured pose",
              "robot_config":str(args.roboenv_dir / "configs/robotnik_arm.json"),
              "units":"m","target_min":np.vstack(list(groups.values())).min(0).tolist(),
              "target_max":np.vstack(list(groups.values())).max(0).tolist()}
    (output / "metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    interactive_html(groups,cloud,base,heading,output)
    static_plot(groups,cloud,base,heading,output,not args.no_show)
    print(json.dumps(metadata,indent=2))
    print("Saved to:",output)


if __name__=="__main__":
    main()
