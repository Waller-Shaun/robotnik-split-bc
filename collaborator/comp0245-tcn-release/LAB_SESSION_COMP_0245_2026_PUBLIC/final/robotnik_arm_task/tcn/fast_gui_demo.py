"""Live GUI in an isolated display process; original DIRECT physics at 1 kHz.

The display never steps, supplies observations, or produces actions. Only
cached measured poses cross into the display process. No prerecorded replay.
"""
from contextlib import contextmanager
import multiprocessing as mp
import time
import traceback
from unittest.mock import patch


def pack_pose(physics, step):
    """Read cached state only: do not call additional PyBullet physics APIs."""
    values = [float(step)]
    for robot in physics.bot:
        for field in (robot.base_position, robot.base_orientation,
                      robot.motor_angles, robot.motor_velocities):
            values.extend(float(value) for value in field)
    return values


def _display_worker(sim_args, sim_kwargs, target, layouts, fps, state, stop,
                    ready, frames, acknowledged):
    from simulation_and_control import pb
    display = None
    try:
        display = pb.SimInterface(*sim_args, **dict(sim_kwargs, use_gui=True))
        client = display.pybullet_client
        if [list(robot.active_joint_ids) for robot in display.bot] != layouts:
            raise ValueError('Display and physics joint layouts differ')
        client.setRealTimeSimulation(0)
        client.configureDebugVisualizer(client.COV_ENABLE_GUI, 0)
        client.resetDebugVisualizerCamera(2.7, 45., -25., [0., -.4, .4])
        visual = client.createVisualShape(client.GEOM_SPHERE, radius=.03,
                                         rgbaColor=[1., 0., 0., .85])
        client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual,
                               basePosition=list(target))
        ready.send(dict(ready=True))
        previous = -1
        while not stop.is_set():
            before = time.perf_counter()
            if not client.isConnected():
                raise RuntimeError('Display window disconnected')
            with state.get_lock():
                snapshot = list(state[:])
            sequence, offset = int(snapshot[0]), 1
            if sequence != previous:
                for robot, joints in zip(display.bot, layouts):
                    n = len(joints)
                    position = snapshot[offset:offset+3]
                    orientation = snapshot[offset+3:offset+7]
                    positions = snapshot[offset+7:offset+7+n]
                    velocities = snapshot[offset+7+n:offset+7+2*n]
                    offset += 7 + 2*n
                    client.resetBasePositionAndOrientation(robot.bot_pybullet,
                                                           position, orientation)
                    client.resetJointStatesMultiDof(
                        robot.bot_pybullet, joints,
                        targetValues=[[q] for q in positions],
                        targetVelocities=[[qd] for qd in velocities])
                previous = sequence
                frames.value += 1
                acknowledged.value = sequence
            stop.wait(max(0., 1./fps - (time.perf_counter()-before)))
    except BaseException:
        try:
            ready.send(dict(ready=False, error=traceback.format_exc()))
        except (OSError, BrokenPipeError):
            pass
        raise
    finally:
        if display is not None and display.pybullet_client.isConnected():
            display.pybullet_client.disconnect()
        ready.close()


class LiveMirror:
    def __init__(self, physics, sim_args, sim_kwargs, target, fps=30,
                 realtime=True):
        if not 1 <= fps <= 120:
            raise ValueError('Display FPS must be between 1 and 120')
        self.physics, self.fps, self.realtime = physics, fps, realtime
        self.dt = float(physics.GetTimeStep())
        self.steps = 0
        self.started = self.previous_step_end = None
        self.next_frame = 0.
        self.physics_s = self.display_s = self.sleep_s = self.between_steps_s = 0.
        context = mp.get_context('spawn')
        self.state = context.Array('d', pack_pose(physics, 0))
        self.stop = context.Event()
        self.frames = context.Value('q', 0)
        self.acknowledged = context.Value('q', -1)
        self.ready, sender = context.Pipe(duplex=False)
        layouts = [list(robot.active_joint_ids) for robot in physics.bot]
        self.process = context.Process(target=_display_worker,
            args=(sim_args, sim_kwargs, list(target), layouts, fps, self.state,
                  self.stop, sender, self.frames, self.acknowledged),
            name='RobotnikLiveDisplay', daemon=True)
        self.process.start()
        sender.close()
        try:
            if not self.ready.poll(30):
                raise TimeoutError('Live GUI did not become ready within 30 seconds')
            status = self.ready.recv()
            if not status.get('ready'):
                raise RuntimeError(status.get('error', 'Live display failed'))
        except BaseException:
            self.close()
            raise

    def publish(self):
        before = time.perf_counter()
        if not self.process.is_alive():
            raise RuntimeError('Live GUI exited; simulation stopped, not silently hidden')
        snapshot = pack_pose(self.physics, self.steps)
        with self.state.get_lock():
            self.state[:] = snapshot
        self.display_s += time.perf_counter()-before

    def step(self, *args, **kwargs):
        before = time.perf_counter()
        if self.started is None:
            self.started = before
        if self.previous_step_end is not None:
            self.between_steps_s += before-self.previous_step_end
        result = self.physics.Step(*args, **kwargs)
        self.physics_s += time.perf_counter()-before
        self.steps += 1
        now = time.perf_counter()
        if now >= self.next_frame:
            self.publish()
            self.next_frame = time.perf_counter()+1./self.fps
        # Absolute deadlines: never add sleep(dt) when already behind real time.
        if self.realtime:
            remaining = self.started+self.steps*self.dt-time.perf_counter()
            if remaining > 0:
                before_sleep = time.perf_counter()
                time.sleep(remaining)
                self.sleep_s += time.perf_counter()-before_sleep
        self.previous_step_end = time.perf_counter()
        return result

    def timings(self):
        wall = time.perf_counter()-self.started if self.started is not None else 0.
        return dict(viewer_mode='live_mirror_separate_process',
                    physics_connection='DIRECT', display_connection='GUI_display_only',
                    display_fps_limit=self.fps, realtime_pacing=self.realtime,
                    physics_steps=self.steps, physics_dt_s=self.dt,
                    simulation_s=self.steps*self.dt, control_loop_wall_s=wall,
                    displayed_frames=self.frames.value, physics_step_wall_s=self.physics_s,
                    pose_publish_wall_s=self.display_s, deadline_sleep_wall_s=self.sleep_s,
                    between_steps_wall_s=self.between_steps_s,
                    display_affects_control=False, pretrained_replay=False)

    def close(self):
        if self.process.is_alive() and self.started is not None:
            self.publish()
            deadline = time.perf_counter()+.25
            while self.acknowledged.value < self.steps and time.perf_counter() < deadline:
                time.sleep(.005)
        self.stop.set()
        self.process.join(timeout=3.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=3.)
        self.ready.close()


class DisplayedSimulation:
    """Delegate all reads/configuration to physics, wrapping Step only."""
    def __init__(self, mirror):
        object.__setattr__(self, 'mirror', mirror)

    def __getattr__(self, name):
        return getattr(self.mirror.physics, name)

    def __setattr__(self, name, value):
        # configure_gripper_fingers installs _SetMotorTorqueByIds dynamically.
        # Forward writes too, or physics would silently lose its finger hold.
        setattr(self.mirror.physics, name, value)

    def Step(self, *args, **kwargs):
        return self.mirror.step(*args, **kwargs)


@contextmanager
def live_gui(target, fps=30, realtime=True):
    """Scope the display adapter to this demo; frozen evaluator stays intact."""
    from simulation_and_control import pb
    original_class = pb.SimInterface
    resources, mirrors = [], []

    def create(*args, **kwargs):
        if kwargs.get('use_gui', True):
            raise ValueError('Live mirror requires evaluator headless=True')
        physics = original_class(*args, **dict(kwargs, use_gui=False))
        resources.append(physics)
        mirror = LiveMirror(physics, args, kwargs, target, fps, realtime)
        mirrors.append(mirror)
        print(f'LIVE GUI: original 1 kHz DIRECT physics; separate display '
              f'process capped at {fps} FPS (not a replay)', flush=True)
        return DisplayedSimulation(mirror)

    try:
        with patch.object(pb, 'SimInterface', create):
            yield mirrors
    finally:
        for mirror in reversed(mirrors):
            mirror.close()
        for sim in reversed(resources):
            client = sim.pybullet_client
            if client.isConnected():
                client.disconnect()
