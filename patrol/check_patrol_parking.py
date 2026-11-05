"""Geometry-only checks: no simulator, model inference, collection or training."""
import unittest
import numpy as np
from patrol_parking_geometry import parking_goal_from_current_base


class RelativeParkingChecks(unittest.TestCase):
    def test_first_origin_task_preserves_legacy_geometry(self):
        for target in ([0.,1.2,1.], [.4,1.2,.75], [-1.2,-.6,1.3]):
            t=np.asarray(target)
            bearing=np.arctan2(t[1],t[0])
            standoff=np.interp(t[2],[.1,.5,1.1,1.25],[.8,.65,.65,.35])
            expected=t[:2]-standoff*np.array([np.cos(bearing),np.sin(bearing)])
            goal,yaw=parking_goal_from_current_base(t,[0.,0.,0.],0.)
            np.testing.assert_allclose(goal,expected,atol=1e-12)
            self.assertAlmostEqual(yaw,bearing)

    def test_horizontal_translation_equivariance(self):
        target=np.array([.4,1.2,1.]);base=np.array([.038,.497,0.])
        shift=np.array([8.,-5.,0.])
        goal,yaw=parking_goal_from_current_base(target,base,1.6)
        moved,moved_yaw=parking_goal_from_current_base(target+shift,base+shift,1.6)
        np.testing.assert_allclose(moved,goal+shift[:2],atol=1e-12)
        self.assertAlmostEqual(moved_yaw,yaw)

    def test_horizontal_rotation_equivariance(self):
        target=np.array([.4,1.2,1.]);base=np.array([.038,.497,0.]);angle=1.1
        r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        goal,yaw=parking_goal_from_current_base(target,base,1.6)
        t=target.copy();b=base.copy();t[:2]=r@target[:2];b[:2]=r@base[:2]
        rotated,rotated_yaw=parking_goal_from_current_base(t,b,1.6+angle)
        np.testing.assert_allclose(rotated,r@goal,atol=1e-12)
        np.testing.assert_allclose([np.cos(rotated_yaw),np.sin(rotated_yaw)],
                                   [np.cos(yaw+angle),np.sin(yaw+angle)],atol=1e-12)

    def test_next_stage_faces_target_from_current_base(self):
        target=np.array([.4,1.2,1.]);base=np.array([.038,.497,0.])
        goal,yaw=parking_goal_from_current_base(target,base,1.6)
        direction=(target[:2]-base[:2])/np.linalg.norm(target[:2]-base[:2])
        np.testing.assert_allclose(target[:2]-goal,.65*direction,atol=1e-12)
        self.assertGreater(np.dot(goal-base[:2],direction),0.)
        self.assertAlmostEqual(yaw,np.arctan2(direction[1],direction[0]))

    def test_coincident_horizontal_target_uses_current_yaw(self):
        goal,yaw=parking_goal_from_current_base([1.,2.,1.],[1.,2.,0.],.8)
        np.testing.assert_allclose(goal,[1.-.65*np.cos(.8),2.-.65*np.sin(.8)])
        self.assertAlmostEqual(yaw,.8)

    def test_invalid_observation_rejected(self):
        with self.assertRaises(ValueError):
            parking_goal_from_current_base([0.,1.,1.],[np.nan,0.,0.],0.)


if __name__=='__main__':unittest.main()
