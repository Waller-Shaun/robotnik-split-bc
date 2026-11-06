"""Map current validation outcomes in 3D, with an optional paired reference batch.

Produces a standalone interactive HTML, PNG/PDF figures, and a failure table.
Reads saved rollouts only; never reruns or edits an evaluation.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROBOENV = Path(r"F:\UCL\COMP0245 Labs\RoboEnv")
DEFAULT_BATCH = ROBOENV / "bc_training_23d_noise_wide_finetune/run_20261031_025012_026265/batch_tests/val_20261031_031404_387244"
DEFAULT_REFERENCE = ROBOENV / "bc_training_23d_noise_wide/run_20261030_112637_239667/batch_tests/val_20261031_022415_233671"
GREEN, RED, BLUE, ORANGE = "#208f79", "#d0444c", "#287cc0", "#dc8a24"


def load_rows(batch):
    with (batch / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["target"] = [float(row["target_" + a]) for a in "xyz"]
        row["success"] = row["success"].lower() == "true"
        for field in ["min_error_m", "final_error_m", "phase_switch_time_s", "end_time_s"]:
            row[field] = float(row[field])
    keys = {tuple(r["target"]) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("Repeated targets in results")
    return rows


def enrich(rows, reference):
    old = {tuple(r["target"]): r for r in reference}
    if old.keys() != {tuple(r["target"]) for r in rows}:
        raise ValueError("Reference and current batch must have identical target sets")
    number = 0
    for row in rows:
        previous = old[tuple(row["target"])]["success"]
        row["change"] = "improved" if row["success"] and not previous else "regressed" if previous and not row["success"] else "unchanged"
        row["failure_id"] = ""
        row["path"] = []
        if not row["success"]:
            number += 1
            row["failure_id"] = f"F{number}"
            with np.load(row["rollout_file"], allow_pickle=False) as data:
                index = int(np.argmin(data["ee_errors"]))
                row["nearest_time_s"] = float(data["timestamps"][index])
                row["nearest_position"] = data["ee_positions"][index].tolist()
                row["final_position"] = data["ee_positions"][-1].tolist()
                indices = np.unique(np.r_[np.linspace(0, len(data["ee_positions"])-1, 700).astype(int), index])
                row["path"] = np.round(data["ee_positions"][indices], 5).tolist()


def initial_robot():
    cache = ROBOENV / "target_visualizations/splits_20261030_initial/target_points.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return data["robot_points"].copy()
    from plot_target_splits import robot_cloud
    return robot_cloud(ROBOENV)[0]


def figures(rows, cloud, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*cloud.T, s=.65, c="#738093", alpha=.38, depthshade=False, rasterized=True)
    for success, color, marker in [(True, GREEN, "o"), (False, RED, "X")]:
        p = np.array([r["target"] for r in rows if r["success"] == success])
        ax.scatter(*p.T, s=36 if success else 110, c=color, marker=marker, depthshade=False,
                   edgecolors="white", linewidths=.5,
                   label=f"{'Success' if success else 'Failure'} ({len(p)})")
    for row in rows:
        p = row["target"]
        if not row["success"]:
            ax.text(p[0]+.015, p[1]+.015, p[2]+.025, row["failure_id"], color="#b52e39", fontsize=9, weight="bold")
    all_points = np.vstack([cloud, [r["target"] for r in rows]])
    lo, hi = all_points.min(0)-.12, all_points.max(0)+.12
    ax.set(xlabel="World X (m)", ylabel="World Y (m)", zlabel="World Z (m)",
           xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]))
    ax.set_box_aspect(hi-lo)
    ax.view_init(elev=25,azim=-55)
    ax.set_title("Validation targets: successes and failures",fontsize=18,pad=20)
    ax.legend(loc="upper left",bbox_to_anchor=(-.02,.97))
    fig.text(.5,.055,"60/70 reached within 1 cm | 20 s horizon | Measurement noise OFF\n"
             "All failures lie at z = 0.3 or 0.4 m. Grey cloud: robot initial collision surface.",ha="center",color="#566074")
    fig.subplots_adjust(left=.02,right=.98,bottom=.12,top=.92)
    for ext in ["png","pdf"]:
        fig.savefig(output/f"failure_map_3d.{ext}",dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(13,5.5),sharex=True,sharey=True)
    for ax,z in zip(axes,[.3,.4]):
        selected=[r for r in rows if abs(r["target"][2]-z)<1e-6]
        for row in selected:
            x,y,_=row["target"]
            ax.scatter(x,y,s=95 if row["success"] else 140,marker="o" if row["success"] else "X",
                       color=GREEN if row["success"] else RED,zorder=3)
            if row["failure_id"]:
                ax.annotate(row["failure_id"],(x,y),xytext=(6,7),textcoords="offset points",color=RED,weight="bold")
        successes=sum(r["success"] for r in selected)
        ax.set_title(f"z = {z:.1f} m | {successes}/{len(selected)} successful",fontsize=13)
        ax.set(xlim=(-.7,.7),ylim=(.84,1.48),xlabel="Target X (m)")
        ax.set_xticks(np.arange(-.6,.61,.2));ax.set_yticks(np.arange(.9,1.41,.1));ax.grid(alpha=.2)
    axes[0].set_ylabel("Target Y (m)")
    fig.suptitle("Low-height validation targets | Green: success; red X: failure",fontsize=15)
    fig.tight_layout(rect=(0,0,1,.92))
    fig.savefig(output/"low_height_slices.png",dpi=200)
    fig.savefig(output/"low_height_slices.pdf")
    plt.close(fig)


def write_html(rows, cloud, output):
    data = []
    for r in rows:
        data.append({k:r[k] for k in ["target","success","change","failure_id","min_error_m","final_error_m","path"]}
                    | {"time":r.get("nearest_time_s"),"nearest":r.get("nearest_position"),"final":r.get("final_position")})
    html = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>验证目标失败位置</title>
<style>*{box-sizing:border-box}body{margin:0;font:14px system-ui,"Microsoft YaHei",sans-serif;background:#f5f7fb;color:#18304c}
header{padding:20px 28px 10px}h1{font-size:24px;margin:0 0 7px}p{color:#62718a;margin:5px 0}
nav{display:flex;flex-wrap:wrap;gap:15px;padding:8px 28px;align-items:center}button,select{padding:6px 10px;border:1px solid #d2dbe7;border-radius:6px;background:white}
canvas{width:100%;height:62vh;min-height:430px;display:block;background:white;touch-action:none}
section{padding:16px 28px}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{border:1px solid #dde4ed;padding:8px;text-align:center}th{background:#eaf0f7}tbody tr{cursor:pointer}tbody tr:hover,tbody tr.active{background:#fff0e9}
#tip{position:fixed;display:none;white-space:pre-line;background:#18304c;color:white;border-radius:6px;padding:9px;pointer-events:none;z-index:3}
#details{min-height:25px;padding:8px 28px;color:#5b5279}label{cursor:pointer}footer{padding:0 28px 20px;color:#62718a;font-size:12px}
</style><header><h1>验证目标空间分布　60 / 70 成功　85.71%</h1>
<p>绿色：成功 · 红色叉：失败 · 蓝圈：微调后改善 · 橙圈：微调后退步</p>
<p>拖动旋转，滚轮缩放；悬停查看坐标。点击失败点或表格行，可查看对应末端轨迹。</p></header>
<nav><label><input id="success" type="checkbox" checked>成功60点</label><label><input id="failure" type="checkbox" checked>失败10点</label>
<label><input id="robot" type="checkbox" checked>初始机器人</label><label><input id="changes" type="checkbox">标出微调变化</label>
<label><input id="labels" type="checkbox" checked>失败编号</label><select id="height"><option value="all">所有高度</option><option value="0.3">仅0.3m</option><option value="0.4">仅0.4m</option></select>
<button id="reset">重置视角</button><button id="top">俯视</button><button id="clear">清除轨迹</button></nav>
<canvas id="view"></canvas><div id="tip"></div><div id="details">当前未选中失败案例。</div>
<section><h2>失败目标　点击一行显示轨迹</h2><table><thead><tr><th>编号</th><th>目标 [x,y,z] m</th><th>最低误差 cm</th><th>最低误差时刻 s</th><th>最终误差 cm</th><th>微调前后</th></tr></thead><tbody id="table"></tbody></table></section>
<footer>仿真20秒内任意时刻误差小于1cm即成功，无保持要求，无8秒门槛。灰色机器人仅表示初始碰撞表面；紫色路径为末端轨迹，黄色点为最近位置。坐标为世界坐标、单位米。</footer>
<script>
const D=__DATA__,robot=__ROBOT__,cv=document.getElementById('view'),ctx=cv.getContext('2d'),tip=document.getElementById('tip');
const $=id=>document.getElementById(id),wrap={improved:'由失败变成功',regressed:'由成功变失败',unchanged:'不变'};
let yaw=-.72,pitch=.43,zoom=1,drag=null,movement=0,selected=null,screen=[];const center=[0,.59,.39];
function project(p){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],u=Math.cos(yaw)*x-Math.sin(yaw)*y,v=Math.sin(yaw)*x+Math.cos(yaw)*y;
let w=Math.cos(pitch)*z-Math.sin(pitch)*v,d=Math.cos(pitch)*v+Math.sin(pitch)*z,s=Math.min(cv.clientWidth/2.5,cv.clientHeight/1.9)*zoom;
return [cv.clientWidth/2+u*s,cv.clientHeight/2-w*s,d]}
function line(a,b,color,width=1){a=project(a);b=project(b);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke()}
function circle(p,r,color){let q=project(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],r,0,Math.PI*2);ctx.fill()}
function draw(){let ratio=devicePixelRatio||1;cv.width=cv.clientWidth*ratio;cv.height=cv.clientHeight*ratio;ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,cv.clientWidth,cv.clientHeight);
for(let i=-4;i<=8;i++)line([i/5,-.6,0],[i/5,1.6,0],'#e7edf4');for(let i=-3;i<=8;i++)line([-.8,i/5,0],[.9,i/5,0],'#e7edf4');
for(const [e,c,n] of [[[.45,0,0],'#bd555f','X'],[[0,.45,0],'#288d73','Y'],[[0,0,.45],'#6079bd','Z']]){line([0,0,0],e,c,2);let q=project(e);ctx.fillStyle=c;ctx.font='13px system-ui';ctx.fillText(n,q[0]+5,q[1]-5)}
if($('robot').checked){ctx.globalAlpha=.4;for(const p of robot)circle(p,1,'#718095');ctx.globalAlpha=1}
if(selected){for(let i=1;i<selected.path.length;i++)line(selected.path[i-1],selected.path[i],'#8060b9',1.5);circle(selected.nearest,5,'#e9ac24');circle(selected.final,5,'#8060b9')}
screen=[];let height=$('height').value;
for(const r of D){if(height!=='all'&&Math.abs(r.target[2]-Number(height))>1e-6)continue;if(!$(r.success?'success':'failure').checked)continue;screen.push({r,q:project(r.target)})}
screen.sort((a,b)=>a.q[2]-b.q[2]);
for(const {r,q} of screen){ctx.lineWidth=2.5;ctx.strokeStyle=r.success?'#208f79':'#d0444c';if(r.success){circle(r.target,4.6,'#208f79')}else{ctx.beginPath();ctx.moveTo(q[0]-5,q[1]-5);ctx.lineTo(q[0]+5,q[1]+5);ctx.moveTo(q[0]-5,q[1]+5);ctx.lineTo(q[0]+5,q[1]-5);ctx.stroke();if($('labels').checked){ctx.fillStyle='#b32f39';ctx.font='bold 12px system-ui';ctx.fillText(r.failure_id,q[0]+8,q[1]-8)}}
if($('changes').checked&&r.change!=='unchanged'){ctx.strokeStyle=r.change==='improved'?'#287cc0':'#dc8a24';ctx.lineWidth=2;ctx.beginPath();ctx.arc(q[0],q[1],9,0,Math.PI*2);ctx.stroke()}
if(r===selected){ctx.strokeStyle='#8060b9';ctx.beginPath();ctx.arc(q[0],q[1],13,0,Math.PI*2);ctx.stroke()}}
ctx.fillStyle='#62718a';ctx.font='12px system-ui';ctx.fillText('世界坐标（m）；地面网格0.2m',15,cv.clientHeight-15)}
function choose(r){selected=r;for(const row of document.querySelectorAll('tbody tr'))row.classList.toggle('active',row.dataset.id===(r?r.failure_id:''));$('details').textContent=r?r.failure_id+'　目标 ['+r.target.join(', ')+'] m　最低误差 '+(r.min_error_m*100).toFixed(3)+' cm　紫色：末端路径；黄色：最近位置':'当前未选中失败案例。';draw()}
function closest(e){let box=cv.getBoundingClientRect(),x=e.clientX-box.left,y=e.clientY-box.top;return screen.map(o=>({o,d:Math.hypot(o.q[0]-x,o.q[1]-y)})).sort((a,b)=>a.d-b.d)[0]}
cv.onpointerdown=e=>{drag=[e.clientX,e.clientY,yaw,pitch];movement=0;cv.setPointerCapture(e.pointerId);tip.style.display='none'};
cv.onpointermove=e=>{if(drag){let dx=e.clientX-drag[0],dy=e.clientY-drag[1];movement=Math.max(movement,Math.hypot(dx,dy));if(movement>3){yaw=drag[2]+dx*.008;pitch=Math.max(-1.55,Math.min(1.55,drag[3]+dy*.008));draw()}return}
let near=closest(e);if(near&&near.d<12){let r=near.o.r;tip.textContent=(r.failure_id||'成功')+'  ['+r.target.join(', ')+'] m\n最低误差 '+(r.min_error_m*100).toFixed(3)+' cm\n最终误差 '+(r.final_error_m*100).toFixed(2)+' cm\n'+wrap[r.change];tip.style.left=Math.min(e.clientX+12,innerWidth-245)+'px';tip.style.top=(e.clientY+12)+'px';tip.style.display='block'}else tip.style.display='none'};
cv.onpointerup=e=>{if(movement<4){let near=closest(e);if(near&&near.d<15&&!near.o.r.success)choose(near.o.r)}drag=null};cv.onpointercancel=()=>drag=null;cv.onpointerleave=()=>tip.style.display='none';
cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.4,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});
for(const id of ['success','failure','robot','changes','labels','height'])$(id).onchange=draw;
$('reset').onclick=()=>{yaw=-.72;pitch=.43;zoom=1;draw()};$('top').onclick=()=>{yaw=0;pitch=Math.PI/2;zoom=1;draw()};$('clear').onclick=()=>choose(null);
for(const r of D.filter(r=>!r.success)){let tr=document.createElement('tr');tr.dataset.id=r.failure_id;tr.innerHTML='<td>'+r.failure_id+'</td><td>['+r.target.map(v=>v.toFixed(1)).join(', ')+']</td><td>'+(r.min_error_m*100).toFixed(3)+'</td><td>'+r.time.toFixed(3)+'</td><td>'+(r.final_error_m*100).toFixed(2)+'</td><td>'+wrap[r.change]+'</td>';tr.onclick=()=>choose(r);$('table').appendChild(tr)}window.onresize=draw;draw();
</script></html>'''
    html=html.replace('__DATA__',json.dumps(data,separators=(',',':'),allow_nan=False)).replace('__ROBOT__',json.dumps(np.round(cloud,4).tolist(),separators=(',',':')))
    (output/'failure_map.html').write_text(html,encoding='utf-8')


def report(rows, batch, reference, output):
    failed=[r for r in rows if not r['success']]
    lines=['# 验证目标失败位置分析','',f'当前批次：`{batch}`。',f'微调前参照：`{reference}`。','',
           '70个验证目标中60个成功、10个失败，成功率85.71%。失败点全部在0.3m和0.4m高度；所有目标都完成了导航切换。',
           '', '[打开交互式三维图](failure_map.html)', '', '![成功与失败目标的三维分布](failure_map_3d.png)', '',
           '绿色为成功，红色叉号为失败；灰色是机器人初始碰撞表面，不是失败时的实际姿态。','',
           '![低高度切片](low_height_slices.png)', '',
           '| 编号 | 目标 [x,y,z] m | 最低误差 cm | 对应时间 s | 最终误差 cm | 微调变化 |',
           '|---|---|---:|---:|---:|---|']
    for r in failed:
        change='由成功变失败' if r['change']=='regressed' else '仍然失败'
        lines.append(f"| {r['failure_id']} | {r['target']} | {r['min_error_m']*100:.3f} | {r['nearest_time_s']:.3f} | {r['final_error_m']*100:.2f} | {change} |")
    lines += ['', '## 观察与解释', '',
              '1. z=0.3m：3/10成功，7个失败；z=0.4m：12/15成功，3个失败；z≥0.5m：45/45成功。',
              '2. 失败点同时出现在负x、零x和正x，不能归结为某一侧x坐标。高度与失败的关联更突出，但目标数较少，不能仅凭图判断因果。',
              '3. 所有失败案例都曾接近至约1.06～2.40cm，最终误差约20～64cm。共同现象是接近后未能进入容差并出现后续偏离。',
              '4. 导航切换全部完成，不代表底盘后续保持静止；空间分布本身不能区分关节力矩、底盘漂移或训练覆盖不足等原因。',
              '5. F5对应此前观察的[0.2,1.1,0.4]，最低误差约1.057cm。可在HTML中点击F5或对应表格行查看末端三维路径。',
              '', '评估条件：20秒上限，测量噪声关闭，任意时刻欧氏距离误差小于1cm即成功，无保持要求。这里只展示验证目标，不能将没有验证点的空间区域解释为成功或失败。']
    (output/'失败位置分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fields=['failure_id','target_x','target_y','target_z','min_error_cm','nearest_time_s','final_error_cm','change']
    with (output/'failure_points.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in failed:
            writer.writerow(dict(zip(fields,[r['failure_id'],*r['target'],r['min_error_m']*100,r['nearest_time_s'],r['final_error_m']*100,r['change']])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir',type=Path,default=DEFAULT_BATCH)
    parser.add_argument('--reference-batch',type=Path,default=DEFAULT_REFERENCE)
    parser.add_argument('--output-dir',type=Path,default=None)
    args=parser.parse_args()
    rows,old=load_rows(args.batch_dir),load_rows(args.reference_batch)
    # Ensure the default report describes the expected complete validation experiment.
    if len(rows)!=70 or sum(r['success'] for r in rows)!=60:
        raise ValueError('This report layout expects the 60/70 validation result')
    enrich(rows,old)
    cloud=initial_robot()
    output=args.output_dir or args.batch_dir/'failure_map'
    output.mkdir(parents=True,exist_ok=False)
    figures(rows,cloud,output);write_html(rows,cloud,output);report(rows,args.batch_dir,args.reference_batch,output)
    print('Figures and report saved to:',output)
    print('Failure IDs:',[(r['failure_id'],r['target']) for r in rows if not r['success']])


if __name__=='__main__':
    main()
