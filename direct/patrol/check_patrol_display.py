"""Display contract checks with a fake debug client; no simulator is started."""
import unittest
import numpy as np
from patrol_route_planner import plan_patrol_route
from patrol_route_display import PatrolRouteDisplay


class DebugClient:
    def __init__(self):self.next_id=0;self.lines=[];self.text=[];self.removed=[]
    def addUserDebugLine(self,start,end,**kw):
        self.next_id+=1;self.lines.append((self.next_id,start,end,kw));return self.next_id
    def addUserDebugText(self,text,position,**kw):
        self.next_id+=1;self.text.append((self.next_id,text,position,kw));return self.next_id
    def removeUserDebugItem(self,item):self.removed.append(item)


class RouteDisplayChecks(unittest.TestCase):
    def setUp(self):
        self.plan=plan_patrol_route([[2.,0.,1.],[.8,0.,1.],[1.2,0.,1.]],[0.,0.,0.],0.)

    def test_planned_route_uses_parking_points_and_labels_visit_order(self):
        client=DebugClient();display=PatrolRouteDisplay(client)
        display.draw_plan(self.plan)
        np.testing.assert_allclose(client.lines[0][1],[0.,0.,.035])
        np.testing.assert_allclose(client.lines[0][2],[.15,0.,.035])
        texts=[x[1] for x in client.text]
        self.assertIn('Visit 1 / Target #2',texts)
        self.assertIn('P3',texts)
        self.assertTrue(all(x[3]['lifeTime']==0 for x in client.lines))

    def test_disabled_display_does_not_call_client(self):
        client=DebugClient();display=PatrolRouteDisplay(client,False)
        display.draw_plan(self.plan);display.show_current_leg([0.,0.,0.],[.2,1.],1)
        self.assertEqual(client.lines,[]);self.assertEqual(client.text,[])

    def test_active_leg_updates_from_observation_without_removing_static_route(self):
        client=DebugClient();display=PatrolRouteDisplay(client);display.draw_plan(self.plan)
        static_ids=set(range(1,client.next_id+1))
        display.show_current_leg([.2,.3,0.],[.6,.7],1)
        old_active=display.active_ids.copy()
        display.show_current_leg([.6,.7,0.],[1.,.9],2)
        self.assertEqual(client.removed,old_active)
        self.assertFalse(static_ids.intersection(client.removed))
        new_line=next(x for x in client.lines if x[0]==display.active_ids[0])
        np.testing.assert_allclose(new_line[1],[.6,.7,.035])
        np.testing.assert_allclose(new_line[2],[1.,.9,.035])
        self.assertEqual(new_line[3]['lineColorRGB'],[1.,.45,.05])


if __name__=='__main__':unittest.main()
