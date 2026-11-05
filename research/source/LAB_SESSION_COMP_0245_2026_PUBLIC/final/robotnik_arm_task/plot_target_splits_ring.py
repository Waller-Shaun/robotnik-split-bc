"""[Ring visualization change] Visualize the actual 931-target ring split.

No rollout or training is performed. HTML is self-contained; PNG/PDF are also saved.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_ROBOENV = Path(r"F:\UCL\COMP0245 Labs\RoboEnv")
DEFAULT_RUN = DEFAULT_ROBOENV / "bc_training_26d_noise_ring"
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


def robot_cloud(config_dir):
    # [Visualization] Read the real robot configuration; never edit it or step physics.
    # Rays sample collision geometry, rather than pretending this is sensor data.
    from simulation_and_control import pb
    sim = pb.SimInterface("robotnik_arm.json", conf_file_path_ext=str(config_dir), use_gui=False)
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
    from matplotlib import font_manager
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "font.size": 11, "axes.unicode_minus": False, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(12, 9), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*cloud.T, s=.7, c="#596575", alpha=.55, depthshade=False,
               label="机器人初始姿态（碰撞表面）", rasterized=True)
    for split, points in groups.items():
        label = {"train": "训练集", "val": "验证集", "test": "测试集"}[split]
        ax.scatter(*points.T, s=14, c=COLORS[split], alpha=.88, depthshade=False,
                   edgecolors="white", linewidths=.25, label=f"{label} ({len(points)})")
    ax.scatter(*base, color="#172b4d", marker="x", s=60, linewidth=2)
    ax.quiver(base[0], base[1], .03, .32*np.cos(heading), .32*np.sin(heading), 0,
              color="#172b4d", linewidth=2, arrow_length_ratio=.25)
    ax.text(base[0]-.1, base[1]-.22, .03, "初始底盘 / 朝向", color="#172b4d", fontsize=9)
    points = np.vstack([cloud, *groups.values()])
    lo, hi = points.min(0)-.12, points.max(0)+.12
    lo[2] = min(0, lo[2])
    ax.set(xlim=(lo[0],hi[0]), ylim=(lo[1],hi[1]), zlim=(lo[2],hi[2]),
           xlabel="世界 X (m)", ylabel="世界 Y (m)", zlabel="世界 Z (m)")
    ax.set_box_aspect(hi-lo)
    ax.view_init(elev=28, azim=-54)
    fig.suptitle("当前环状数据集：931个目标与机器人初始姿态", fontsize=19, y=.96)
    ax.legend(loc="upper left", bbox_to_anchor=(-.04,1.01), fontsize=10, framealpha=.96)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
    fig.text(.5,.035,"水平半径 0—2 m · 7个高度层：0.20 / 0.35 / 0.50 / 0.75 / 1.00 / 1.15 / 1.30 m\n"
             "固定初始位姿；旧单侧1014条未合并。灰色为仿真碰撞表面采样，非深度相机实测。",
             ha="center", color="#566074", fontsize=10)
    fig.subplots_adjust(left=.02,right=.99,bottom=.1,top=.9)
    fig.savefig(output / "target_splits.png", dpi=220)
    fig.savefig(output / "target_splits.pdf")
    # [Ring visualization change] 只画一个高度层的俯视图，避免7层XY点重叠。
    top_fig, top_ax = plt.subplots(figsize=(9, 9), facecolor="white")
    top_ax.scatter(cloud[:, 0], cloud[:, 1], s=.8, c="#667587", alpha=.45, rasterized=True)
    height = .35
    for split, points in groups.items():
        layer = points[np.isclose(points[:, 2], height)]
        label = {"train": "训练集", "val": "验证集", "test": "测试集"}[split]
        top_ax.scatter(layer[:, 0], layer[:, 1], s=31, c=COLORS[split], edgecolors="white",
                       linewidths=.5, label=f"{label} ({len(layer)})")
    top_ax.arrow(base[0], base[1], .5*np.cos(heading), .5*np.sin(heading), color="#172b4d",
                 width=.015, head_width=.08, length_includes_head=True)
    top_ax.text(base[0]-.22, base[1]-.34, "初始机器人", color="#172b4d")
    top_ax.set(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), xlabel="世界 X (m)", ylabel="世界 Y (m)",
               title="单层俯视：z = 0.35 m，133个目标")
    top_ax.set_aspect("equal"); top_ax.grid(alpha=.18); top_ax.legend(loc="upper left")
    top_fig.tight_layout()
    top_fig.savefig(output / "target_splits_top_z0p35.png", dpi=200)
    plt.close(top_fig)
    if show:
        plt.show()
    else:
        plt.close(fig)


def interactive_html(groups, cloud, base, heading, output):
    data = {k: np.round(v,6).tolist() for k,v in groups.items()}
    heights = sorted(set(float(z) for z in np.vstack(list(groups.values()))[:, 2]))
    data.update(robot=np.round(cloud,4).tolist(), base=base.tolist(), heading=float(heading),
                heights=heights, counts={k:len(v) for k,v in groups.items()}, center=[0,0,.65])
    html = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>931目标环状数据集 | 三维空间分布</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f7fb;color:#172b4d;font:15px system-ui,"Microsoft YaHei",sans-serif}
header{padding:24px 30px 12px}h1{font-size:25px;margin:0 0 8px}p{margin:6px 0;color:#62718a}
nav{padding:8px 30px;display:flex;gap:20px;align-items:center;flex-wrap:wrap}
label{cursor:pointer}button{padding:7px 14px;background:white;border:1px solid #d8dfeb;border-radius:8px;cursor:pointer}
canvas{display:block;width:100%;height:calc(100vh - 260px);min-height:480px;background:#fff;touch-action:none}
#tip{position:fixed;display:none;background:#172b4d;color:white;border-radius:6px;padding:8px;pointer-events:none;font-size:13px}
footer{padding:10px 30px;color:#62718a;font-size:12px}
</style><header><h1>当前环状数据集：931个目标与机器人初始姿态</h1>
<p>半径 0—2 m · 7个高度层（0.20—1.30 m）· 固定初始位姿 · 旧单侧1014条未合并</p>
<p>拖动旋转 · 滚轮缩放 · 悬停查看目标坐标 · 按高度筛选；图中点为目标位置，并非运动轨迹。</p></header>
<nav><label style="color:#2674bf"><input type="checkbox" id="train" checked> 训练集 <span id="trainCount"></span></label>
<label style="color:#df8b16"><input type="checkbox" id="val" checked> 验证集 <span id="valCount"></span></label>
<label style="color:#b83778"><input type="checkbox" id="test" checked> 测试集 <span id="testCount"></span></label>
<label><input type="checkbox" id="robot" checked> 机器人初始姿态</label>
<label>高度 <select id="height"><option value="all">全部高度</option></select></label>
<button id="reset">重置视角</button><button id="top">俯视图</button><button id="front">侧视图</button></nav>
<canvas id="view"></canvas><div id="tip"></div>
<footer>世界坐标，单位米，各轴使用相同尺度。灰色机器人点云采自配置初始姿态的仿真碰撞表面，非真实深度扫描。初始朝向以深色箭头表示。</footer>
<script>
const D=__DATA__,cv=document.getElementById('view'),ctx=cv.getContext('2d'),tip=document.getElementById('tip');
const colors={train:'#2674bf',val:'#df8b16',test:'#b83778',robot:'#667587'};
let yaw=-.72,pitch=.58,zoom=1,drag=null,moved=false,screen=[];
const center=D.center;
const heightSelect=document.getElementById('height');
for(const h of D.heights){let op=document.createElement('option');op.value=String(h);op.textContent=h.toFixed(2)+' m';heightSelect.appendChild(op)}
function pointsAtHeight(key){return heightSelect.value==='all'?D[key]:D[key].filter(p=>Math.abs(p[2]-Number(heightSelect.value))<1e-5)}
function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];
let u=Math.cos(yaw)*x-Math.sin(yaw)*y,v=Math.sin(yaw)*x+Math.cos(yaw)*y;
let w=Math.cos(pitch)*z-Math.sin(pitch)*v,d=Math.cos(pitch)*v+Math.sin(pitch)*z;
let s=Math.min(cv.clientWidth/5.0,cv.clientHeight/3.8)*zoom;
return [cv.clientWidth/2+u*s,cv.clientHeight/2-w*s,d]}
function line(a,b,color,width=1){a=project(a);b=project(b);ctx.strokeStyle=color;ctx.lineWidth=width;
ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke()}
function draw(){const ratio=window.devicePixelRatio||1;cv.width=cv.clientWidth*ratio;cv.height=cv.clientHeight*ratio;
ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,cv.clientWidth,cv.clientHeight);
for(let i=-11;i<=11;i++){let x=i/5;line([x,-2.2,0],[x,2.2,0],'#edf0f5');line([-2.2,x,0],[2.2,x,0],'#edf0f5');}
let visibleHeights=heightSelect.value==='all'?D.heights:[Number(heightSelect.value)];
for(const h of visibleHeights)for(let i=0;i<64;i++){let a=2*Math.PI*i/64,b=2*Math.PI*(i+1)/64;line([2*Math.cos(a),2*Math.sin(a),h],[2*Math.cos(b),2*Math.sin(b),h],'#e3e8ef',.6)}
for(const [end,color,name] of [[[.45,0,0],'#c65f61','X'],[[0,.45,0],'#438b77','Y'],[[0,0,.45],'#657ac9','Z']]){
line([0,0,0],end,color,2);let q=project(end);ctx.fillStyle=color;ctx.font='13px system-ui';ctx.fillText(name,q[0]+5,q[1]-5)}
for(const key of ['train','val','test'])document.getElementById(key+'Count').textContent='('+pointsAtHeight(key).length+')';
let all=[];for(const key of ['robot','train','val','test'])if(document.getElementById(key).checked)
(key==='robot'?D.robot:pointsAtHeight(key)).forEach(p=>all.push({key,p,q:project(p)}));all.sort((a,b)=>a.q[2]-b.q[2]);screen=[];
for(const o of all){let r=o.key==='robot'?1:4.4;ctx.globalAlpha=o.key==='robot'?.58:.95;ctx.fillStyle=colors[o.key];
ctx.beginPath();ctx.arc(o.q[0],o.q[1],r,0,Math.PI*2);ctx.fill();
if(o.key!=='robot'){ctx.strokeStyle='#fff';ctx.lineWidth=.6;ctx.stroke();screen.push(o)}}ctx.globalAlpha=1;
if(document.getElementById('robot').checked){let a=[D.base[0],D.base[1],.02],b=[a[0]+.32*Math.cos(D.heading),a[1]+.32*Math.sin(D.heading),.02];
line(a,b,'#172b4d',3);let q=project([D.base[0]-.2,D.base[1]-.35,.02]);ctx.fillStyle='#172b4d';ctx.fillText('初始机器人 / +X朝向',q[0],q[1]);}
ctx.fillStyle='#62718a';ctx.fillText('世界坐标 (m) | 地面网格 0.2 m | '+(heightSelect.value==='all'?'全部7层':'高度 '+heightSelect.value+' m'),18,cv.clientHeight-18)}
cv.onpointerdown=e=>{drag=[e.clientX,e.clientY,yaw,pitch];moved=false;cv.setPointerCapture(e.pointerId);tip.style.display='none'};
cv.onpointermove=e=>{if(drag){let dx=e.clientX-drag[0],dy=e.clientY-drag[1];moved=true;yaw=drag[2]+dx*.008;pitch=Math.max(-1.55,Math.min(1.55,drag[3]+dy*.008));draw();return;}
let rect=cv.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;
let near=screen.map(o=>({o,d:Math.hypot(o.q[0]-x,o.q[1]-y)})).sort((a,b)=>a.d-b.d)[0];
if(near&&near.d<10){const names={train:'训练集',val:'验证集',test:'测试集'};let r=Math.hypot(near.o.p[0]-D.base[0],near.o.p[1]-D.base[1]);tip.textContent=names[near.o.key]+' ['+near.o.p.map(v=>v.toFixed(3)).join(', ')+'] m；水平半径 '+r.toFixed(2)+' m';tip.style.left=Math.max(4,Math.min(e.clientX+12,innerWidth-390))+'px';tip.style.top=(e.clientY+12)+'px';tip.style.display='block'}else tip.style.display='none'};
cv.onpointerup=cv.onpointercancel=()=>drag=null;cv.onpointerleave=()=>tip.style.display='none';
cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.4,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});
for(const key of ['train','val','test','robot'])document.getElementById(key).onchange=draw;
heightSelect.onchange=()=>{tip.style.display='none';draw()};
document.getElementById('reset').onclick=()=>{yaw=-.72;pitch=.58;zoom=1;draw()};
document.getElementById('top').onclick=()=>{yaw=0;pitch=Math.PI/2;zoom=1;draw()};window.onresize=draw;draw();
document.getElementById('front').onclick=()=>{yaw=0;pitch=0;zoom=1;draw()};
</script></html>'''
    (output / "target_splits.html").write_text(html.replace("__DATA__",json.dumps(data,separators=(",",":"))),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=DEFAULT_RUN / "data_split.json")
    # [Ring visualization change] 与采集统一读取项目配置，不改其噪声开关。
    parser.add_argument("--config-dir",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir",type=Path,default=None)
    parser.add_argument("--no-show",action="store_true",help="Save files without opening Matplotlib")
    args=parser.parse_args()
    groups=load_targets(args.manifest)
    if {k:len(v) for k,v in groups.items()} != {"train":651,"val":140,"test":140}:
        raise ValueError("This visualization expects the current 931-target ring manifest")
    cloud,base,heading=robot_cloud(args.config_dir)
    output=args.output_dir or DEFAULT_ROBOENV / "target_visualizations" / datetime.now().strftime("ring_%Y%m%d_%H%M%S")
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output / "target_points.npz",**groups,robot_points=cloud,base_position=base,base_heading=heading)
    metadata={"manifest":str(args.manifest.resolve()),"counts":{k:len(v) for k,v in groups.items()},
              "robot_points":len(cloud),"robot_geometry":"collision surface, ray sampled, initial configured pose",
              "robot_config":str(args.config_dir / "configs/robotnik_arm.json"),
              "manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "robot_config_sha256":hashlib.sha256((args.config_dir / "configs/robotnik_arm.json").read_bytes()).hexdigest(),
              "heights_m":sorted(set(float(v) for v in np.vstack(list(groups.values()))[:,2])),
              "radii_m":sorted(set(float(v) for v in np.round(np.linalg.norm(np.vstack(list(groups.values()))[:,:2]-base[:2],axis=1),4))),
              "source_observation_dim":23,"current_trained_input_dim":26,"planned_stage_input_dim":29,
              "stage_29d_implemented":False,"old_single_side_1014_merged":False,
              "units":"m","target_min":np.vstack(list(groups.values())).min(0).tolist(),
              "target_max":np.vstack(list(groups.values())).max(0).tolist()}
    (output / "metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    interactive_html(groups,cloud,base,heading,output)
    static_plot(groups,cloud,base,heading,output,not args.no_show)
    print(json.dumps(metadata,indent=2))
    print("Saved to:",output)


if __name__=="__main__":
    main()
