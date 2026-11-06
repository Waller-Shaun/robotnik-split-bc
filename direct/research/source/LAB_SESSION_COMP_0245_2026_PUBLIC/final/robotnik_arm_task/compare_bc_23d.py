"""Compare one BC rollout with an expert episode for the same target.

Offline predictions use expert observations, not rollout observations.
Closed-loop curves have different states at equal simulation times.
This script only reads model/data files and writes diagnostic outputs.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import only the architecture; importing does not start training.
from train_bc_23d import BCPolicy


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def phase_time(times, observations):
    indices = np.flatnonzero(observations[:, 22] == 0)
    return float(times[indices[0]]) if len(indices) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_run = (
        Path(__file__).resolve().parents[3] / "RoboEnv" / "bc_training_23d"
        / "run_20261028_204007_921743"
    )
    parser.add_argument("--run-dir", type=Path, default=default_run)
    parser.add_argument("--rollout", type=Path)
    parser.add_argument("--expert", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=10.0)
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("--end must be greater than --start")

    run_dir = args.run_dir.resolve()
    if args.rollout is None:
        candidates = list((run_dir / "rollouts").glob("*.npz"))
        if not candidates:
            raise FileNotFoundError("No rollouts found; run test_bc_23d.py first")
        rollout_path = max(candidates, key=lambda p: p.stat().st_mtime)
    else:
        rollout_path = args.rollout.resolve()
    rollout = read_npz(rollout_path)
    target = rollout["target"]
    manifest = json.loads((run_dir / "data_split.json").read_text(encoding="utf-8"))

    # Match by target coordinates, since filenames contain recording times.
    if args.expert is None:
        matches = []
        for split, filenames in manifest["splits"].items():
            for filename in filenames:
                path = Path(manifest["data_dir"]) / filename
                with np.load(path, allow_pickle=False) as data:
                    if np.allclose(data["target"], target, rtol=0, atol=1e-6):
                        matches.append((path, split))
        if len(matches) != 1:
            raise ValueError(f"Found {len(matches)} matching experts; specify --expert")
        expert_path, expert_split = matches[0]
    else:
        expert_path, expert_split = args.expert.resolve(), "explicit_file"
    expert = read_npz(expert_path)
    if not np.allclose(expert["target"], target, rtol=0, atol=1e-6):
        raise ValueError("Expert and rollout targets differ")

    ex, ey, et = expert["observations"], expert["actions"], expert["timestamps"]
    rx, ra = rollout["observations"], rollout["actions"]
    # Rollout timestamps include a final state without an associated action.
    rt = rollout["timestamps"][:len(rx)]
    if ex.shape != (len(et), 23) or ey.shape != (len(et), 11):
        raise ValueError("Expected aligned 23D expert inputs and 11D actions")
    if rx.shape != (len(ra), 23) or ra.shape[1:] != (11,):
        raise ValueError("Expected aligned 23D rollout inputs and 11D actions")

    torch.set_num_threads(1)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model = BCPolicy()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    norm = read_npz(run_dir / "normalization.npz")
    pred = np.empty_like(ey)
    with torch.inference_mode():
        for start in range(0, len(ex), 4096):
            end = start + 4096
            x = ((ex[start:end] - norm["x_mean"]) / norm["x_std"]).astype(np.float32)
            output = model(torch.from_numpy(x)).numpy()
            # No clipping: measure the network's actual prediction error.
            pred[start:end] = output * norm["y_std"] + norm["y_mean"]

    output_dir = run_dir / "diagnostics" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    ep, rp = phase_time(et, ex), phase_time(rt, rx)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def decorate(ax):
        if ep is not None:
            ax.axvline(ep, color="#2563eb", ls=":", alpha=0.6)
        if rp is not None:
            ax.axvline(rp, color="#ea580c", ls=":", alpha=0.6)
        ax.set_xlim(args.start, args.end)
        ax.grid(alpha=0.2)

    def panels(filename, title, labels, curves, ylabel):
        fig, axs = plt.subplots(int(np.ceil(len(labels) / 2)), 2,
                                figsize=(13, 2.4 * int(np.ceil(len(labels) / 2))),
                                squeeze=False, constrained_layout=True)
        for i, label in enumerate(labels):
            ax = axs.flat[i]
            for times, values, name, color, style in curves:
                ax.plot(times, values[:, i], label=name, color=color, ls=style, lw=1.2)
            ax.set_title(label)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Simulation time (s)")
            decorate(ax)
        for ax in list(axs.flat)[len(labels):]:
            ax.set_visible(False)
        axs.flat[0].legend(fontsize=8)
        fig.suptitle(title + "\nDotted lines: expert / rollout navigation completion", fontsize=12)
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    # Same-time closed-loop curves describe behavior, NOT matched-state errors.
    panels("01_joint_angles.png", "Joint angles: separate closed-loop trajectories",
           [f"J{i+1}" for i in range(7)], [
               (et, ex[:, 5:12], "Expert measured", "#2563eb", "-"),
               (rt, rx[:, 5:12], "BC measured", "#ea580c", "--")], "Angle (rad)")
    curves = [
        (et, ey, "Expert action", "#2563eb", "-"),
        (et, pred, "BC on expert states (unclipped)", "#16a34a", "--"),
        (rt, ra, "BC rollout applied (clipped)", "#ea580c", "-")]
    panels("02_wheel_commands.png", "Wheel commands: offline fit and closed-loop behavior",
           ["Front right", "Front left", "Back left", "Back right"],
           [(t, v[:, :4], n, c, s) for t, v, n, c, s in curves], "Speed (rad/s)")
    panels("03_arm_torques.png", "Arm torques: offline fit and closed-loop behavior",
           [f"J{i+1}" for i in range(7)],
           [(t, v[:, 4:], n, c, s) for t, v, n, c, s in curves], "Torque (N m)")

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    axs[0].plot(rollout["timestamps"], rollout["ee_errors"], color="#ea580c", label="BC EE error")
    axs[0].axhline(0.01, color="red", ls="--", label="1 cm")
    axs[0].set_ylabel("Distance (m)")
    axs[0].legend()
    for i, label in enumerate(["x", "y"]):
        axs[1].plot(et, ex[:, i], label=f"Expert base {label}")
        axs[1].plot(rt, rx[:, i], ls="--", label=f"BC base {label}")
    axs[1].set_ylabel("Position (m)")
    axs[1].legend()
    axs[2].step(et, ex[:, 22], label="Expert phase", where="post")
    axs[2].step(rt, rx[:, 22], label="BC phase", where="post", ls="--")
    axs[2].set_ylabel("base_active")
    axs[2].legend()
    for ax in axs:
        decorate(ax)
        ax.set_xlabel("Simulation time (s)")
    fig.savefig(output_dir / "04_error_base_phase.png", dpi=160)
    plt.close(fig)

    # RMSE uses matched expert states and all samples, regardless of plot window.
    with (output_dir / "offline_rmse.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["phase", "samples", "action", "unit", "rmse"])
        for phase, mask in [("all", np.ones(len(ex), dtype=bool)),
                            ("navigation", ex[:, 22] == 1), ("arm_reach", ex[:, 22] == 0)]:
            if not mask.any():
                continue
            rmse = np.sqrt(np.mean((pred[mask] - ey[mask]).astype(np.float64)**2, axis=0))
            for i, value in enumerate(rmse):
                writer.writerow([phase, int(mask.sum()), f"wheel_{i+1}" if i < 4 else f"arm_J{i-3}",
                                 "rad/s" if i < 4 else "N m", float(value)])
    summary = {"run_dir": str(run_dir), "rollout": str(rollout_path),
               "expert": str(expert_path), "expert_split": expert_split,
               "target": target.tolist(), "expert_phase_time": ep, "bc_phase_time": rp,
               "checkpoint_epoch": checkpoint["epoch"],
               "plot_window_s": [args.start, args.end],
               "note": "Offline RMSE is matched-state; closed-loop time curves are not matched-state comparisons."}
    (output_dir / "sources.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Target: {target}; expert split: {expert_split}")
    print(f"Expert phase switch: {ep}; BC phase switch: {rp}")
    print(f"Plots and offline RMSE saved to: {output_dir}")


if __name__ == "__main__":
    main()
