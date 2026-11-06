"""Route planning checks only: no simulation, policy inference or training."""
import itertools
import unittest
import numpy as np
from patrol_route_planner import plan_patrol_route


def reference_cost(targets, order, start, yaw):
    """Independent brute-force reference, including order-dependent stopping geometry."""
    xy=np.array(start[:2],dtype=float);total=0.
    for i in order:
        t=np.asarray(targets[i]);d=t[:2]-xy;r=np.linalg.norm(d)
        u=d/r if r>1e-8 else np.array([np.cos(yaw),np.sin(yaw)])
        stop=t[:2]-np.interp(t[2],[.1,.5,1.1,1.25],[.8,.65,.65,.35])*u
        total+=np.linalg.norm(stop-xy);xy=stop;yaw=np.arctan2(u[1],u[0])
    return float(total)


class RoutePlanningChecks(unittest.TestCase):
    def test_unordered_line_targets_have_hand_calculated_shortest_route(self):
        p=plan_patrol_route([[2.,0.,1.],[.8,0.,1.],[1.2,0.,1.]],[0.,0.,0.],0.)
        self.assertEqual(p['visit_order_input_ids'],[2,3,1])
        self.assertAlmostEqual(p['input_order_estimated_distance_m'],1.85)
        self.assertAlmostEqual(p['selected_order_estimated_distance_m'],1.35)
        self.assertTrue(p['exact_optimal_for_estimated_cost'])
        self.assertFalse(p['actual_route_optimality_claimed'])

    def test_exact_search_matches_independent_exhaustive_reference(self):
        targets=np.array([[1.1,.2,.5],[-.9,.6,1.3],[.2,-1.4,1.15],[.8,1.1,.75]])
        start=np.array([.15,-.1,.02]);yaw=.4
        expected=min(reference_cost(targets,o,start,yaw) for o in itertools.permutations(range(4)))
        p=plan_patrol_route(targets,start,yaw)
        self.assertAlmostEqual(p['selected_order_estimated_distance_m'],expected,places=10)
        self.assertEqual(sorted(p['visit_order_input_ids']),[1,2,3,4])

    def test_given_mode_preserves_indices_and_cost(self):
        p=plan_patrol_route([[2.,0.,1.],[.8,0.,1.],[1.2,0.,1.]],[0.,0.,0.],0.,'given')
        self.assertEqual(p['visit_order_input_ids'],[1,2,3])
        self.assertEqual(p['estimated_saving_m'],0.)
        self.assertFalse(p['exact_optimal_for_estimated_cost'])

    def test_rigid_horizontal_transform_preserves_route_and_cost(self):
        targets=np.array([[1.1,.2,.5],[-.9,.6,1.3],[.2,-1.4,1.15],[.8,1.1,.75]])
        base=np.array([.15,-.1,.02]);yaw=.4;angle=1.2;shift=np.array([9.,-4.])
        r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        a=plan_patrol_route(targets,base,yaw)
        moved=targets.copy();moved[:,:2]=targets[:,:2]@r.T+shift
        b=base.copy();b[:2]=r@base[:2]+shift
        transformed=plan_patrol_route(moved,b,yaw+angle)
        self.assertEqual(a['visit_order_input_ids'],transformed['visit_order_input_ids'])
        self.assertAlmostEqual(a['selected_order_estimated_distance_m'],transformed['selected_order_estimated_distance_m'])

    def test_duplicate_targets_and_zero_length_keep_distinct_input_ids(self):
        p=plan_patrol_route([[.65,0.,1.]]*3,[0.,0.,0.],0.)
        self.assertEqual(p['visit_order_input_ids'],[1,2,3])
        self.assertEqual(p['selected_order_estimated_distance_m'],0.)
        self.assertEqual(p['estimated_saving_percent'],0.)

    def test_large_route_is_labelled_heuristic_and_never_worse_than_given(self):
        targets=[[float(x),.8,1.] for x in [1.8,-.7,1.4,-1.,.8,1.1,-.9,1.6,.5]]
        p=plan_patrol_route(targets,[0.,0.,0.],0.)
        self.assertFalse(p['exact_optimal_for_estimated_cost'])
        self.assertEqual(sorted(p['visit_order_input_ids']),list(range(1,10)))
        self.assertLessEqual(p['selected_order_estimated_distance_m'],p['input_order_estimated_distance_m']+1e-12)

    def test_planning_does_not_modify_input_arrays(self):
        targets=np.array([[1.,.8,1.],[-1.,.8,1.]])
        base=np.array([.2,.1,0.]);old_targets=targets.copy();old_base=base.copy()
        plan_patrol_route(targets,base,.3)
        np.testing.assert_array_equal(targets,old_targets);np.testing.assert_array_equal(base,old_base)

    def test_bad_targets_are_rejected(self):
        for targets in ([],[[1.,2.]],[[np.nan,0.,1.]]):
            with self.assertRaises(ValueError):plan_patrol_route(targets,[0.,0.,0.],0.)


if __name__=='__main__':unittest.main()
