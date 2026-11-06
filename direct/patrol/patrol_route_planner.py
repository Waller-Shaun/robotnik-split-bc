"""[巡检排序改动] 优化预计停车路程，不生成机器人控制动作。

顺序会影响下一停车点，不能把目标间距离简单预存成固定TSP距离矩阵。
每个候选顺序都递推停车几何；不返回起点，不优化转向时间或机械臂时间。
"""
import numpy as np
from patrol_parking_geometry import parking_goal_from_current_base


ROUTE_PLANNER_VERSION = 'open_route_sequential_relative_parking_v1'
EXACT_MAX_TARGETS = 8
TWO_OPT_MAX_PASSES = 10
EPS = 1e-12


def _next_stop(target, base, yaw):
    goal, heading = parking_goal_from_current_base(target, base, yaw)
    distance = float(np.linalg.norm(goal - base[:2]))
    # [改动] 用预计停车位置作为下一候选段起点；离地高度保持初始值。
    return np.r_[goal, base[2]], heading, distance


def _evaluate(targets, order, start, yaw):
    base=start.copy(); total=0.; legs=[]
    for index in order:
        next_base, next_yaw, distance = _next_stop(targets[index], base, yaw)
        legs.append({'input_target_index':int(index)+1, 'target':targets[index].tolist(),
                     'estimated_start_base_xyz':base.tolist(),
                     'estimated_parking_xy':next_base[:2].tolist(),
                     'estimated_final_heading_rad':float(next_yaw),
                     'estimated_horizontal_distance_m':distance})
        total+=distance; base=next_base; yaw=next_yaw
    return float(total), legs


def plan_patrol_route(targets, observed_start_base, observed_start_yaw, mode='shortest'):
    """返回JSON可序列化计划。输入编号从1开始，内部排列索引从0开始。

    <=8点：分支限界穷举，保证本名义停车成本最小；并不保证实际最短。
    >8点：贪心起点加开放路线2-opt，保留输入顺序作为候选对照；仅近似。
    """
    targets=np.asarray(targets,dtype=np.float64)
    start=np.asarray(observed_start_base,dtype=np.float64)
    if (targets.ndim!=2 or targets.shape[1]!=3 or len(targets)==0
            or start.shape!=(3,) or not np.isfinite(targets).all()
            or not np.isfinite(start).all() or not np.isfinite(observed_start_yaw)):
        raise ValueError('Expected nonempty finite targets (N,3) and observed base XYZ/yaw')
    if mode not in ('shortest','given'):
        raise ValueError('Route mode must be shortest or given')
    yaw=float(observed_start_yaw); n=len(targets); input_order=tuple(range(n))
    input_cost,_=_evaluate(targets,input_order,start,yaw)
    best_order=input_order;best_cost=input_cost
    evaluated=1;pruned=0;search_nodes=0;passes=0
    exact=False

    if mode=='given':
        method='input_order'
    elif n<=EXACT_MAX_TARGETS:
        method='exact_branch_and_bound';exact=True

        def search(prefix, remaining, base, heading, cost):
            nonlocal best_order,best_cost,evaluated,pruned,search_nodes
            search_nodes+=1
            if not remaining:
                evaluated+=1
                if cost<best_cost-EPS:
                    best_order=prefix;best_cost=cost
                return
            # 所有剩余距离非负；当前累计已不优则无需展开。
            if cost>=best_cost-EPS:
                pruned+=1
                return
            for index in remaining:
                next_base,next_yaw,distance=_next_stop(targets[index],base,heading)
                search(prefix+(index,),tuple(i for i in remaining if i!=index),
                       next_base,next_yaw,cost+distance)

        search((),input_order,start,yaw,0.)
    else:
        method='greedy_and_open_two_opt_heuristic'
        remaining=list(input_order);greedy=[];base=start.copy();heading=yaw
        while remaining:
            options=[(_next_stop(targets[i],base,heading),i) for i in remaining]
            (base,heading,_),index=min(options,key=lambda item:(item[0][2],item[1]))
            greedy.append(index);remaining.remove(index)
        greedy=tuple(greedy);greedy_cost,_=_evaluate(targets,greedy,start,yaw);evaluated+=1
        # 同时从输入顺序及贪心顺序改进，最终一定不会比输入顺序成本更大。
        for seed in (input_order,greedy):
            current=seed;current_cost,_=_evaluate(targets,current,start,yaw);evaluated+=1
            for _ in range(TWO_OPT_MAX_PASSES):
                passes+=1;improved_order=current;improved_cost=current_cost
                for i in range(n-1):
                    for j in range(i+1,n):
                        candidate=current[:i]+tuple(reversed(current[i:j+1]))+current[j+1:]
                        # 不能用固定边差公式：翻转顺序后必须重算后续停车点。
                        cost,_=_evaluate(targets,candidate,start,yaw);evaluated+=1
                        if cost<improved_cost-EPS:
                            improved_order=candidate;improved_cost=cost
                if improved_order==current:break
                current=improved_order;current_cost=improved_cost
            if current_cost<best_cost-EPS:
                best_order=current;best_cost=current_cost

    cost,legs=_evaluate(targets,best_order,start,yaw)
    saving=max(0.,input_cost-cost)
    return {
        'planner_version':ROUTE_PLANNER_VERSION,'requested_mode':mode,'algorithm':method,
        'cost_definition':'sum_of_estimated_base_to_parking_XY_distances',
        'return_to_start':False,'turning_time_in_cost':False,'arm_time_in_cost':False,
        'obstacles_considered':False,'world_truth_used_for_planning':False,
        'observed_start_base_xyz':start.tolist(),'observed_start_yaw_rad':yaw,
        'input_targets':targets.tolist(),'input_order_ids':list(range(1,n+1)),
        'visit_order_indices_zero_based':list(best_order),
        'visit_order_input_ids':[int(i)+1 for i in best_order],
        'execution_targets':targets[list(best_order)].tolist(),
        'input_order_estimated_distance_m':input_cost,
        'selected_order_estimated_distance_m':cost,
        'estimated_saving_m':saving,
        'estimated_saving_percent':100.*saving/input_cost if input_cost>EPS else 0.,
        'exact_optimal_for_estimated_cost':exact,
        'actual_route_optimality_claimed':False,
        'exact_max_targets':EXACT_MAX_TARGETS,'evaluated_complete_orders':evaluated,
        'pruned_prefixes':pruned,'search_nodes':search_nodes,'two_opt_passes':passes,
        'estimated_legs':legs,
        'execution_note':'Plan once; recompute each actual stage parking goal from its actual observed starting pose.'
    }
