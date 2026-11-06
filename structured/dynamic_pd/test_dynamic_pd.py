import unittest
import numpy as np
import torch
from .common import load_fixed
from .model import GainNetwork,supervised_torque
from .controller import DynamicPDController
from robotnik_arm_control import BaseNavState


class GainTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_exact_initial_gains_and_bounds(self):
        model = GainNetwork()
        x = torch.randn(10,128)
        gains,delta = model(x)
        self.assertTrue(torch.equal(gains,model.center.expand(10,-1)))
        self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))
        with torch.no_grad():
            model.net[-1].weight.normal_(0,100)
        gains,_ = model(x)
        self.assertTrue(((gains[:,:7]>=80)&(gains[:,:7]<=120)).all())
        self.assertTrue(((gains[:,7:]>=24)&(gains[:,7:]<=36)).all())

    def test_torque_gradients_and_saturation(self):
        model = GainNetwork()
        gains,_ = model(torch.randn(8,128))
        torque = supervised_torque(gains,torch.full((8,7),.01),torch.full((8,7),.02),
                                  torch.zeros(8,7),torch.zeros(8,7))
        torque.square().mean().backward()
        self.assertTrue(torch.isfinite(model.net[-1].weight.grad).all())
        self.assertGreater(model.net[-1].weight.grad.abs().sum(),0)
        clipped = supervised_torque(gains,torch.ones(8,7)*100,torch.zeros(8,7),torch.zeros(8,7),torch.zeros(8,7))
        self.assertTrue(torch.equal(clipped,torch.ones(8,7)*30))

    def test_network_can_depend_on_state(self):
        model = GainNetwork()
        with torch.no_grad():
            model.net[-1].weight.normal_(0,.1)
        a = model(torch.zeros(1,128))[0]
        b = model(torch.ones(1,128))[0]
        self.assertFalse(torch.equal(a,b))

    def test_zero_head_matches_original_actions_and_reset(self):
        old = load_fixed()
        new = DynamicPDController(zero_head=True)
        rng = np.random.default_rng(42)
        observations = []
        for tick in range(300):
            obs = np.zeros(26,dtype=np.float32)
            obs[4] = 1
            obs[5:12] = rng.normal(0,.1,7)
            obs[12:19] = rng.normal(0,.01,7)
            obs[19:22] = [-.9,1.2,1.1]
            obs[22] = tick<150
            obs[23:26] = rng.normal(0,.02,3)
            context = dict(base_position=np.zeros(3),base_orientation=np.array([0.,0.,0.,1.]),
                bearing=0.,goal_position=np.array([-.51,.68]),goal_bearing=2.2143,
                phase=BaseNavState(phase='align' if tick<150 else 'done',done=tick>=150))
            np.testing.assert_array_equal(old.predict(obs,**context),new.predict(obs,**context))
            observations.append((obs,context))
        self.assertEqual(new.gain_tick,300)
        self.assertTrue((new.gain_state==[100.]*7+[30.]*7).all())
        new.reset()
        old.reset()
        self.assertEqual(len(new.history),0)
        self.assertIsNone(new.q_ref)
        self.assertEqual(new.gain_tick,0)
        self.assertEqual(len(new.gain_trace),0)
        obs,context = observations[0]
        np.testing.assert_array_equal(old.predict(obs,**context),new.predict(obs,**context))


if __name__ == '__main__':
    unittest.main()
