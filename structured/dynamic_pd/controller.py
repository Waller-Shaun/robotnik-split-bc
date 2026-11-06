"""State-dependent gain policy. Online references never use expert labels."""
import json
import numpy as np
import torch
from .common import BASE, ROOT, sha
from test_structured_arm import StructuredController
from structured_arm import features, decode
from robotnik_navigation_policy import build_navigation_observation
from .model import GainNetwork


class DynamicPDController(StructuredController):
    def __init__(self, checkpoint=None, zero_head=False):
        super().__init__(BASE / 'models/tcn', 'best_model.pt')
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.gain_model = GainNetwork().eval()
        if not zero_head:
            checkpoint = checkpoint or ROOT / 'runs/full/best_model.pt'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if saved['schema_version'] != 1:
                raise ValueError('Wrong dynamic gain checkpoint schema')
            if saved['base_checkpoint_sha256'] != sha(self.checkpoint_path):
                raise ValueError('Motion model differs from gain training')
            for name in ('dynamic_pd/model.py','dynamic_pd/controller.py'):
                if sha(ROOT/name)!=saved['config']['source_sha256'][name]:
                    raise ValueError('Gain inference source changed since training')
            self.gain_model.load_state_dict(saved['gain_state_dict'])
            if not all(torch.isfinite(v).all() for v in self.gain_model.state_dict().values()):
                raise ValueError('Nonfinite gain checkpoint')
            self.gain_checkpoint = str(checkpoint)
        self.policy_type = 'supervised_structured_TCN_state_dependent_PD'
        self.provenance.pop('fixed_tracking_gains', None)
        self.provenance.update(experiment='supervised_dynamic_pd',
            gain_bounds={'p': [80.,120.], 'd': [24.,36.]},
            gain_filter_time_constant_s=.010, gains_learned_from_expert_torque=True,
            motion_and_navigation_weights_frozen=True, gain_labels_not_fabricated=True,
            is_hybrid_controller=True, not_unrestricted_direct_torque_network=True)
        self.reset()

    def reset(self):
        super().reset()
        self.gain_state = np.array([100.] * 7 + [30.] * 7, dtype=np.float64)
        self.gain_tick = 0
        self.gain_trace = []
        self.gain_min = np.full(14, np.inf)
        self.gain_max = np.full(14, -np.inf)
        self.gain_sum = np.zeros(14)
        self.gain_sum_sq = np.zeros(14)
        self.max_gain_step = np.zeros(14)

    def predict(self, observation, **context):
        raw = features(observation, context['base_orientation'])
        self.history.append(raw)
        frames = np.asarray(self.history)
        history = frames[np.maximum(len(frames)-1-np.arange(19,-1,-1)*10,0)]
        x = np.ascontiguousarray((history-self.stats['x_mean'])/self.stats['x_std'])
        nav_x = build_navigation_observation(context['base_position'], context['bearing'],
                     context['goal_position'], context['goal_bearing'], context['phase'])
        with torch.inference_mode():
            hidden = self.model.encoder(torch.from_numpy(x)[None].transpose(1,2))[:,:,-1]
            output = self.model.head(hidden)
            velocity, gravity = decode(output, torch.from_numpy(raw)[None], self.g_mean, self.g_std)
            velocity, gravity = velocity[0].numpy().copy(), gravity[0].numpy()
            gain_request = self.gain_model(hidden)[0][0].numpy().astype(np.float64)
            wheels = self.nav(torch.from_numpy((nav_x-self.nav_stats['x_mean'])/self.nav_stats['x_std'])[None])[0].numpy()
        q = np.asarray(observation[5:12], dtype=np.float64)
        qd = np.asarray(observation[12:19], dtype=np.float64)
        if self.q_ref is None:
            self.q_hold = q.copy()
            self.q_ref = q.copy()
        if observation[22] > .5:
            self.q_ref = self.q_hold.copy()
            velocity = np.zeros(7)
        else:
            blocked = ((q<=self.lower)&(velocity<0)) | ((q>=self.upper)&(velocity>0))
            velocity[blocked] = 0
            self.q_ref = np.clip(self.q_ref+.001*velocity, q-.15, q+.15)
            self.q_ref = np.clip(self.q_ref, self.lower, self.upper)
        before = self.gain_state.copy()
        self.gain_state += (.001 / (.010 + .001)) * (gain_request-self.gain_state)
        kp, kd = self.gain_state[:7], self.gain_state[7:]
        # Full regressor output is joint torque; this PD structure is disclosed.
        torque = kp*(self.q_ref-q) + kd*(velocity-qd) + gravity
        action = np.r_[wheels*self.nav_stats['y_std']+self.nav_stats['y_mean'], torque]
        if not np.isfinite(action).all():
            raise FloatingPointError('Invalid dynamic-PD action')
        if (kp<80-1e-6).any() or (kp>120+1e-6).any() or (kd<24-1e-6).any() or (kd>36+1e-6).any():
            raise ValueError('Applied gain outside predeclared bounds')
        self.gain_min = np.minimum(self.gain_min, self.gain_state)
        self.gain_max = np.maximum(self.gain_max, self.gain_state)
        self.gain_sum += self.gain_state
        self.gain_sum_sq += self.gain_state**2
        self.max_gain_step = np.maximum(self.max_gain_step, abs(self.gain_state-before))
        if self.gain_tick % 10 == 0:
            self.gain_trace.append(np.r_[self.gain_tick*.001, self.gain_state, gain_request])
        self.gain_tick += 1
        return action

    def gain_summary(self):
        if self.gain_tick == 0:
            center = [100.]*7+[30.]*7
            return dict(control_ticks=0,minimum=center,maximum=center,mean=center,std=[0.]*14,
                        maximum_change_per_1ms=[0.]*14,trace_interval_s=.010,bounds_satisfied=True)
        n = max(self.gain_tick,1)
        mean = self.gain_sum/n
        return dict(control_ticks=self.gain_tick, minimum=self.gain_min.tolist(),
                    maximum=self.gain_max.tolist(), mean=mean.tolist(),
                    std=np.sqrt(np.maximum(self.gain_sum_sq/n-mean**2,0)).tolist(),
                    maximum_change_per_1ms=self.max_gain_step.tolist(),
                    trace_interval_s=.010, bounds_satisfied=True)
