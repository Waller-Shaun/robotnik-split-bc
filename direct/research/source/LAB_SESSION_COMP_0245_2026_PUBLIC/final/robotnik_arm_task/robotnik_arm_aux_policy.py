"""[B2 change] 无GRU、多任务机械臂：26维输入；辅助参考仅作训练标签。"""
from pathlib import Path
import hashlib
import json
import torch

from arm_lowz_supplement_experiment import resolve_finetuned_pair as resolve_baseline_pair

VARIANT = "B2_mlp_arm_torque_aux_reference_delta_v1"
RUN_ROOT = "bc_training_arm_ring_r08_aux_b2"
FROZEN_FILES = ("best_model.pt", "normalization.npz", "config.json", "data_split.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AuxiliaryArmPolicy(torch.nn.Module):
    """[B2 change] 保留旧network参数名，辅助分支共享最后128维隐藏特征。"""
    def __init__(self, hidden_dims=(256, 256, 128)):
        super().__init__()
        dims = [26, *hidden_dims, 11]
        layers = []
        for i, (n, m) in enumerate(zip(dims, dims[1:])):
            layers.append(torch.nn.Linear(n, m))
            if i < len(dims)-2:
                layers.append(torch.nn.ReLU())
        self.network = torch.nn.Sequential(*layers)
        self.aux_head = torch.nn.Linear(hidden_dims[-1], 7)
        # 初始力矩严格继承基线；辅助标准化预测初始为0，即训练标签均值。
        torch.nn.init.zeros_(self.aux_head.weight)
        torch.nn.init.zeros_(self.aux_head.bias)

    def forward_tasks(self, x):
        features = x
        for layer in list(self.network.children())[:-1]:
            features = layer(features)
        return self.network[-1](features), self.aux_head(features)

    def forward(self, x):
        # [B2 change] 部署只执行原力矩路径，辅助输出既不输入控制器也不替代动作。
        return self.network(x)


def load_policy_from_checkpoint(checkpoint):
    c = checkpoint["config"]
    if (c.get("experiment_variant") != VARIANT or c.get("input_dim") != 26
            or c.get("output_dim") != 11 or c.get("auxiliary_output_dim") != 7
            or c.get("expert_references_used_as_input") is not False
            or c.get("gru_used") is not False):
        raise ValueError("Expected B2 MLP+aux checkpoint; no GRU or reference inputs")
    model = AuxiliaryArmPolicy(c["hidden_dims"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("Nonfinite model parameters")
    return model


def resolve_finetuned_pair(roboenv, arm_run=None, nav_run=None):
    if arm_run is None:
        candidates = []
        for p in (Path(roboenv)/RUN_ROOT).glob("run_*"):
            if not (p/"training_finished.json").is_file() or not (p/"best_model.pt").is_file():
                continue
            c = json.loads((p/"config.json").read_text(encoding="utf-8"))
            if c.get("experiment_variant") == VARIANT and not c.get("smoke_only", True):
                candidates.append(p)
        if not candidates:
            raise FileNotFoundError("No completed B2 run: train separately, or pass --arm-run-dir")
        arm_run = sorted(candidates)[-1]
    arm_run = Path(arm_run).resolve()
    c = json.loads((arm_run/"config.json").read_text(encoding="utf-8"))
    if (c.get("experiment_variant") != VARIANT or c.get("training_mode") != "arm_only_aux_multitask_navigation_frozen"
            or c.get("navigation_weights_updated") is not False or c.get("expert_references_used_as_input") is not False):
        raise ValueError("Wrong B2 experiment schema")
    parent, nav, _, nc = resolve_baseline_pair(roboenv, c["source_model_run_dir"], nav_run)
    if str(nav) != c["nav_run_dir"]:
        raise ValueError("Wrong frozen navigation binding")
    for field, folder in [("source_model_file_sha256", parent), ("frozen_nav_file_sha256", nav)]:
        if set(c[field]) != set(FROZEN_FILES):
            raise ValueError("Incomplete source hashes")
        for name, expected in c[field].items():
            if sha256(folder/name) != expected:
                raise ValueError(f"Frozen source changed: {folder/name}")
    for name in ["normalization.npz", "data_split.json"]:
        if sha256(arm_run/name) != sha256(parent/name):
            raise ValueError(f"B2 changed inherited {name}")
    for name, expected in c["inference_module_sha256"].items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Inference module changed: {name}")
    if sha256(arm_run/"aux_normalization.npz") != c["aux_normalization_sha256"]:
        raise ValueError("Auxiliary training statistics changed")
    return arm_run, nav, c, nc
