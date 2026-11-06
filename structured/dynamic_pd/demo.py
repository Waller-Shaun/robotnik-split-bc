"""Run fixed or learned dynamic gains without changing either model."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,BASE,load_fixed
from .controller import DynamicPDController
from rollout_noisy import rollout,NOISY_CONFIG


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=('fixed','dynamic'),default='dynamic')
    p.add_argument('--target',type=float,nargs=3,default=[-.9,1.2,1.1])
    p.add_argument('--seed',type=int,default=137)
    p.add_argument('--headless',action='store_true')
    a = p.parse_args()
    if not np.isfinite(a.target).all() or not 0<=a.seed<2**32:
        p.error('Target must be finite and seed must be uint32')
    torch.set_num_threads(1)
    controller = load_fixed() if a.variant=='fixed' else DynamicPDController()
    np.random.seed(a.seed);torch.manual_seed(a.seed)
    a.duration=20.;a.sensor_noise='config';a.sim_config=NOISY_CONFIG
    a.roboenv_dir=BASE/'external/RoboEnv'
    a.result_file=ROOT/'outputs/demos'/datetime.now(timezone.utc).strftime(a.variant+'_%Y%m%d_%H%M%S_%f.npz')
    if a.headless:
        rollout(a,controller)
    else:
        from fast_gui_demo import live_gui
        a.headless=True
        with live_gui(a.target):rollout(a,controller)
    if a.variant=='dynamic':
        from .common import write
        write(a.result_file.with_suffix('.gains.json'),controller.gain_summary())
        print('Applied gain ranges:',controller.gain_min,controller.gain_max)


if __name__=='__main__':main()
