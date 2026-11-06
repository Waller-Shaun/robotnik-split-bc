"""[路线显示改动] 纯显示层：不创建碰撞物体、不读取传感器、不参与控制。"""
import numpy as np


class PatrolRouteDisplay:
    """青色预定停车路线、编号目标；橙色为当前实际重算的停车段。"""
    def __init__(self, client, enabled=True):
        self.client=client;self.enabled=enabled;self.active_ids=[]

    def _line(self,start,end,color,width=3):
        return self.client.addUserDebugLine(list(start),list(end),lineColorRGB=color,
                                            lineWidth=width,lifeTime=0)

    @staticmethod
    def _ground(xy):return [float(xy[0]),float(xy[1]),.035]

    def _arrow(self,start,end,color):
        a=np.asarray(self._ground(start));b=np.asarray(self._ground(end))
        ids=[self._line(a,b,color)]
        delta=b[:2]-a[:2];length=np.linalg.norm(delta)
        if length>1e-8:
            u=delta/length;v=np.array([-u[1],u[0]])
            size=min(.12,.25*length)
            for side in (-1,1):
                tip=b.copy();tip[:2]=b[:2]-size*u+side*.45*size*v
                ids.append(self._line(b,tip,color,2))
        return ids

    def _cross(self,xy,color):
        x,y=map(float,xy);z=.045;d=.055
        return [self._line([x-d,y,z],[x+d,y,z],color),
                self._line([x,y-d,z],[x,y+d,z],color)]

    def draw_plan(self,plan):
        if not self.enabled:return
        color=[.05,.65,.95];start=plan['observed_start_base_xyz'][:2]
        for visit,leg in enumerate(plan['estimated_legs'],1):
            stop=leg['estimated_parking_xy'];self._arrow(start,stop,color);self._cross(stop,color)
            self.client.addUserDebugText(f'P{visit}',[stop[0],stop[1],.12],textColorRGB=color,textSize=1.2,lifeTime=0)
            target=np.asarray(leg['target']).copy();target[2]+=.10
            self.client.addUserDebugText(f"Visit {visit} / Target #{leg['input_target_index']}",
                target.tolist(),textColorRGB=[.1,.1,.1],textSize=1.2,lifeTime=0)
            start=stop

    def show_current_leg(self,observed_base,actual_parking,visit):
        if not self.enabled:return
        for item in self.active_ids:self.client.removeUserDebugItem(item)
        color=[1.,.45,.05]
        self.active_ids=self._arrow(observed_base[:2],actual_parking,color)+self._cross(actual_parking,color)
        self.active_ids.append(self.client.addUserDebugText(f'Active stop {visit}',
            [float(actual_parking[0]),float(actual_parking[1]),.22],
            textColorRGB=color,textSize=1.2,lifeTime=0))


def save_route_preview(out,plan):
    """开始运动前保存独立预定路线图；headless也可查看。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,7))
    xy=np.asarray([plan['observed_start_base_xyz'][:2]]+
                   [leg['estimated_parking_xy'] for leg in plan['estimated_legs']])
    targets=np.asarray(plan['execution_targets'])
    ax.plot(xy[:,0],xy[:,1],'o--',color='deepskyblue',label='Estimated base parking route')
    ax.scatter(*xy[0],marker='s',s=80,color='black',label='Observed start')
    ax.scatter(targets[:,0],targets[:,1],marker='*',s=100,color='crimson',label='EE targets (not base stops)')
    for visit,leg in enumerate(plan['estimated_legs'],1):
        a,b=xy[visit-1],xy[visit]
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'deepskyblue'})
        ax.annotate(f'P{visit}',b,xytext=(5,-12),textcoords='offset points')
        target=leg['target']
        ax.annotate(f"{visit}: #{leg['input_target_index']} (z={target[2]:.2f}m)",target[:2],
                    xytext=(5,7),textcoords='offset points')
    ax.set(xlabel='World X (m)',ylabel='World Y (m)',
           title=f"Planned open route: {plan['selected_order_estimated_distance_m']:.2f} m estimated")
    ax.set_aspect('equal',adjustable='datalim');ax.margins(.18);ax.grid(alpha=.3);ax.legend()
    fig.tight_layout();fig.savefig(out/'route_plan.png',dpi=160);plt.close(fig)
