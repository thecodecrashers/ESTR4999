# maml_reinforce_recovering.py
# ------------------------------------------------------------
#  Meta-Learning (FOMAML) + REINFORCE for Whittle Index
#  Environment: recoveringBanditsEnv  (single recovering arm)
# ------------------------------------------------------------
from __future__ import annotations

import os, json, random, argparse, copy
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import trange

from rc_env import recoveringBanditsEnv   # ← 你的环境文件名若不同请改这里


# ------------------------------------------------------------
#           A c t o r   N e t
# ------------------------------------------------------------
"""def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(*size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, state_dim: int = 1, embed_dim: int = 3,
                 hidden_dim: int = 256, out_init_w: float = 3e-3):
        super().__init__()
        self.in_dim = state_dim + embed_dim          # (s, θ0, θ1, θ2)
        self.fc1 = nn.Linear(self.in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU()
        self._init(out_init_w)

    def _init(self, out_init_w):
        for m in (self.fc1, self.fc2, self.fc3):
            m.weight.data = fanin_init(m.weight.data.size())
            m.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-out_init_w, out_init_w)
        self.fc_out.bias.data.zero_()

    def forward(self, state: torch.Tensor, env_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, env_emb], dim=-1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.fc_out(x)            # (B,1)
"""
class Actor(nn.Module):
    def __init__(self, state_dim: int=1, embed_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = state_dim + embed_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)  # λ̂(s) logit
        self.act = nn.ReLU()

    def forward(self, state: torch.Tensor, env_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, env_emb], dim=-1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.fc_out(x)


# ------------------------------------------------------------
#     R E I N F O R C E   (单条轨迹计算损失，无梯截断)
# ------------------------------------------------------------
def reinforce_loss(
    *,
    actor: nn.Module,
    env: recoveringBanditsEnv,
    device: torch.device,
    lam_val: float,
    gamma: float,
    max_steps: int,
    batch_states: List[int],
    epsilon: float = 0.0,
    force_random: bool = False,
) -> torch.Tensor:
    """返回一个 scalar Tensor（轨迹平均损失）"""
    losses = []
    env_emb_const = torch.tensor(
        [env.theta0, env.theta1, env.theta2], dtype=torch.float32, device=device
    ).unsqueeze(0)

    for s0 in batch_states:
        log_p, rewards = [], []
        env.reset()
        env.arm[0] = s0   # 直接设置等待时间

        for _ in range(max_steps):
            s = torch.tensor([env.arm[0]], dtype=torch.float32,
                             device=device).unsqueeze(0)  # (1,1)
            logit = actor(s, env_emb_const)              # λ̂(s)
            p1 = torch.sigmoid(logit - lam_val)          # π(a=1|s)
            probs = torch.cat([1 - p1, p1], dim=-1).squeeze(0)  # (2,)
            dist = torch.distributions.Categorical(probs=probs)

            if force_random or random.random() < epsilon:
                a = torch.tensor(random.randint(0, 1), device=device)
            else:
                a = dist.sample()

            log_p.append(dist.log_prob(a))
            _, r, done, _ = env.step(a.item())
            if a.item() == 1:     # 拉动成本
                r -= lam_val
            rewards.append(r)
            if done:
                break

        # —— 折扣回报 G_t —— #
        R = 0.0; returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # —— 归一化 —— #
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        adv = returns
        traj_loss = -torch.sum(torch.stack(log_p) * adv)
        losses.append(traj_loss)

    return torch.stack(losses).mean()     # scalar Tensor


# ------------------------------------------------------------
#         W h i t t l e   I n d e x   T a b l e
# ------------------------------------------------------------
@torch.no_grad()
def get_whittle_table(actor: nn.Module, env: recoveringBanditsEnv,
                      device: torch.device) -> List[float]:
    env_emb = torch.tensor([env.theta0, env.theta1, env.theta2],
                           dtype=torch.float32, device=device).unsqueeze(0)
    tbl = []
    for s in range(1, env.maxWait + 1):
        st = torch.tensor([s], dtype=torch.float32,
                          device=device).unsqueeze(0)
        tbl.append(float(actor(st, env_emb).item()))
    return tbl


# ------------------------------------------------------------
#                       M a i n
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # ========= 任务 / 环境 =========
    ap.add_argument("--theta_file", default="theta_pairs_recovering.json",
                    help="JSON 内含 theta_pairs 及 maxWait 字段")
    ap.add_argument("--num", type=int, default=10, help="参与元训练的臂数")
    ap.add_argument("--noise_var", type=float, default=0.05)
    ap.add_argument("--max_wait", type=int, default=None)
    # ========= MAML =========
    ap.add_argument("--meta_iterations", type=int, default=2000)
    ap.add_argument("--tasks_per_meta_batch", type=int, default=5)
    ap.add_argument("--adapt_steps", type=int, default=3)
    ap.add_argument("--adapt_lr", type=float, default=1e-4)
    ap.add_argument("--meta_lr", type=float, default=1e-4)
    # ========= RL =========
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epsilon", type=float, default=0.05)
    # ========= λ =========
    ap.add_argument("--lam_low", type=float, default=0)
    ap.add_argument("--lam_high", type=float, default=10.0)
    # ========= 杂项 =========
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_root", default=None)

    args = ap.parse_args()

    # reproducibility
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- 载入 θ 参数 ----
    raw = json.load(open(args.theta_file, "r", encoding="utf-8"))
    theta_pairs = raw["theta_pairs"][: args.num]
    if len(theta_pairs) < args.num:
        raise ValueError("θ-pairs fewer than requested num")

    max_wait = args.max_wait or raw.get("maxWait", 20)
    save_root = args.save_root or f"maml_reinforce_recovering_{args.num}"
    os.makedirs(save_root, exist_ok=True)

    # 预采样 λ 序列（方便复现）
    total_eps = args.meta_iterations * args.adapt_steps
    lam_seq = [random.uniform(args.lam_low, args.lam_high) for _ in range(total_eps)]
    lam_ptr = 0

    # ---- 构建所有 Task ----
    tasks = []
    for t0, t1, t2 in theta_pairs:
        env = recoveringBanditsEnv(
            seed=args.seed, thetaVals=(t0, t1, t2),
            noiseVar=args.noise_var, maxWait=max_wait
        )
        tasks.append(env)

    # ---- 初始化 meta-actor ----
    meta_actor = Actor().to(device)
    meta_opt = optim.Adam(meta_actor.parameters(), lr=args.meta_lr)
    meta_loss_hist = []

    # ========================================================
    #            M E T A   T R A I N I N G   L O O P
    # ========================================================
    for it in trange(args.meta_iterations, desc="Meta-train"):
        meta_opt.zero_grad()
        # 采样一个 meta-batch 任务
        task_ids = random.sample(range(args.num), args.tasks_per_meta_batch)
        meta_batch_loss_val = 0.0

        for tid in task_ids:
            env = tasks[tid]

            # ---- 1) clone meta-actor ----
            actor_clone = copy.deepcopy(meta_actor)
            clone_opt = optim.SGD(actor_clone.parameters(), lr=args.adapt_lr)

            # ---- 2) 内循环自适应 (K 次) ----
            for k in range(args.adapt_steps):
                batch_states = random.sample(
                    list(range(1, env.maxWait + 1)), args.batch_size
                )
                lam_val = lam_seq[lam_ptr % total_eps]; lam_ptr += 1
                loss_inner = reinforce_loss(
                    actor=actor_clone, env=env, device=device,
                    lam_val=lam_val, gamma=args.gamma,
                    max_steps=args.max_steps, batch_states=batch_states,
                    epsilon=args.epsilon,
                    force_random=(k == 0)           # 第一次完全探索
                )
                clone_opt.zero_grad(); loss_inner.backward(); clone_opt.step()

            # ---- 3) meta-loss on fresh traj —— no 2nd grad ----
            eval_states = random.sample(
                list(range(1, env.maxWait + 1)), args.batch_size
            )
            lam_val = lam_seq[lam_ptr % total_eps]; lam_ptr += 1
            meta_loss = reinforce_loss(
                actor=actor_clone, env=env, device=device,
                lam_val=lam_val, gamma=args.gamma,
                max_steps=args.max_steps, batch_states=eval_states,
                epsilon=0.0, force_random=False
            )
            meta_batch_loss_val += meta_loss.item()

            # ---- 4) accumulate 1st-order grads ----
            grads = torch.autograd.grad(meta_loss, actor_clone.parameters())
            for p_meta, g in zip(meta_actor.parameters(), grads):
                if p_meta.grad is None:
                    p_meta.grad = g.clone().detach()
                else:
                    p_meta.grad += g.clone().detach()

        # ---- 平均梯度 & 更新 meta-params ----
        for p in meta_actor.parameters():
            p.grad /= args.tasks_per_meta_batch
        meta_opt.step()
        meta_loss_hist.append(meta_batch_loss_val / args.tasks_per_meta_batch)

    # ===== 保存 meta-actor & 损失图 =====
    torch.save(meta_actor.state_dict(), os.path.join(save_root, "meta_actor.pt"))
    plt.figure(); plt.plot(meta_loss_hist)
    plt.xlabel("Meta-iteration"); plt.ylabel("Meta-batch loss")
    plt.tight_layout(); plt.savefig(os.path.join(save_root, "meta_loss.png"), dpi=200)
    plt.close()

    # ===== 评估：输出每臂 Whittle 曲线 =====
    all_tables = []
    for idx, env in enumerate(tasks):
        tbl = get_whittle_table(meta_actor, env, device)
        arm_tag = f"arm{idx:02d}"
        json.dump(
            {"theta0": env.theta0, "theta1": env.theta1,
             "theta2": env.theta2, "whittle": tbl},
            open(os.path.join(save_root, f"{arm_tag}_whittle.json"), "w"),
            indent=2
        )
        all_tables.append({"arm": idx, "theta0": env.theta0,
                           "theta1": env.theta1, "theta2": env.theta2,
                           "whittle": tbl})

    json.dump(all_tables, open(os.path.join(save_root, "all_whittle_tables.json"), "w"), indent=2)

    # ---- 画所有臂曲线 ----
    plt.figure(figsize=(6,4))
    for d in all_tables:
        plt.plot(d["whittle"],
                 label=f"A{d['arm']}:θ0={d['theta0']:.2f},θ1={d['theta1']:.2f}")
    plt.xlabel("State s"); plt.ylabel("λ̂(s)")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, "whittle_tables.png"), dpi=250)
    plt.close()

    print("✓ Meta-training done → results in", save_root)


if __name__ == "__main__":
    main()
