"""[B3 change] 只修改共享梯度的辅助贡献；不通过gate继续求导，不干预机器人。"""
import torch

GRADIENT_FIELDS = ["valid_batches", "batch_count", "cosine_mean_valid",
                   "conflict_fraction_valid", "gate_active_fraction", "mean_gate",
                   "mean_effective_aux_weight", "main_shared_grad_norm",
                   "aux_shared_grad_norm", "mean_effective_aux_to_main_norm_ratio"]


def gradient_geometry(main, auxiliary, weight, mode):
    """main/aux为对应共享参数的梯度列表，余弦在整个共享主干上计算。"""
    if mode not in ["gated", "fixed", "off"]:
        raise ValueError("Unknown gradient mode")
    with torch.no_grad():
        m = torch.cat([v.detach().reshape(-1) for v in main])
        a = torch.cat([v.detach().reshape(-1) for v in auxiliary])
        mn, an = torch.linalg.vector_norm(m), torch.linalg.vector_norm(a)
        valid = (mn > 1e-12) & (an > 1e-12)
        cosine = torch.where(valid, (m @ a)/(mn*an).clamp_min(1e-24), torch.zeros_like(mn)).clamp(-1, 1)
        if mode == "gated":
            gate = cosine.clamp_min(0)
        elif mode == "fixed":
            gate = torch.ones_like(cosine)
        else:
            gate = torch.zeros_like(cosine)
        effective = weight*gate
        ratio = torch.where(mn > 1e-12, effective*an/mn.clamp_min(1e-12), torch.zeros_like(mn))
        return {"valid":valid,"cosine":cosine,"gate":gate,"effective":effective,
                "main_norm":mn,"aux_norm":an,"ratio":ratio}


def apply_gradient_update(model, main_loss, auxiliary_loss, optimizer, weight, mode):
    """[B3 change] 两次求梯度：主干门控；主力矩头只看主任务，辅助头独立学习。"""
    layers = list(model.network.children())
    shared = tuple(p for layer in layers[:-1] for p in layer.parameters())
    action_head = tuple(layers[-1].parameters())
    auxiliary_head = tuple(model.aux_head.parameters())
    mg = torch.autograd.grad(main_loss, shared+action_head, retain_graph=True)
    ag = torch.autograd.grad(auxiliary_loss, shared+auxiliary_head)
    ns = len(shared)
    geometry = gradient_geometry(mg[:ns], ag[:ns], weight, mode)
    optimizer.zero_grad(set_to_none=True)
    for p, m, a in zip(shared, mg[:ns], ag[:ns]):
        p.grad = m + geometry["effective"]*a
    for p, m in zip(action_head, mg[ns:]):
        p.grad = m
    # 即便共享梯度被关掉，辅助头仍按lambda学习，避免无法学出非零共享梯度。
    head_weight = weight if mode != "off" else 0.0
    for p, a in zip(auxiliary_head, ag[ns:]):
        p.grad = head_weight*a
    optimizer.step()
    return geometry


def diagnostic_vector(g):
    """每批只在GPU累加，epoch结束再复制到CPU；valid排除零范数批次。"""
    valid = g["valid"].to(torch.float64)
    one = torch.ones_like(valid)
    return torch.stack([valid, one, g["cosine"]*valid,
        (g["cosine"] < 0).to(torch.float64)*valid,
        (g["effective"] > 0).to(torch.float64),g["gate"],g["effective"],
        g["main_norm"],g["aux_norm"],g["ratio"]]).double()


def diagnostic_summary(sums):
    v = sums.detach().cpu().numpy()
    valid, batches = int(v[0]), int(v[1])
    result = {"valid_batches":valid,"batch_count":batches}
    for i, name in enumerate(GRADIENT_FIELDS[2:], 2):
        denom = valid if i in [2, 3] else batches
        result[name] = float(v[i]/denom) if denom else 0.0
    return result
