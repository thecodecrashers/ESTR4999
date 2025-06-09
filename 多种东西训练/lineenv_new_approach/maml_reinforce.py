
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

from lineEnv import lineEnv  # ↖️ 你的 RMAB 单臂环境

# ------------------------------
# Actor 网络
# ------------------------------
class Actor(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.input_dim = state_dim + 3  # state(1) + (p,q,optX)
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)  # 输出 Whittle logit λ̂(s)
        )

    def forward(self, state, env_emb):
        x = torch.cat([state, env_emb], dim=-1)
        return self.net(x)  # (B,1)

# ------------------------------
# REINFORCE loss on a *batch* of starting states
# ------------------------------

def reinforce_loss(
    actor: nn.Module,
    env: lineEnv,
    device: torch.device,
    lam_value: float,
    gamma: float,
    max_steps: int,
    batch_states: List[int],
    epsilon: float = 0.0,
    force_random: bool = False,
):
    losses = []
    for s0 in batch_states:
        log_probs, rewards = [], []
        env.reset(); env.state = s0
        for _ in range(max_steps):
            s = torch.tensor([env.state], dtype=torch.float32, device=device).unsqueeze(0)
            env_emb = torch.tensor([env.p, env.q, float(env.OptX)], dtype=torch.float32, device=device).unsqueeze(0)
            logit = actor(s, env_emb)          # λ̂(s)
            p1 = torch.sigmoid(logit - lam_value)
            probs = torch.cat([1 - p1, p1], dim=-1).squeeze(0)
            dist = torch.distributions.Categorical(probs=probs)
            if force_random or random.random() < epsilon:
                a = torch.tensor(random.randint(0, 1), device=device)
            else:
                a = dist.sample()
            log_probs.append(dist.log_prob(a))
            _, r, done, _ = env.step(a.item())
            if a.item() == 1:
                r -= lam_value
            rewards.append(r)
            if done:
                break
        # 折扣回报
        R = 0.0; returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        # 归一化提升稳定性
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantage = returns
        traj_loss = -torch.sum(torch.stack(log_probs) * advantage)
        losses.append(traj_loss)
    return torch.stack(losses).mean()

# ------------------------------
# 生成整张 Whittle 表
# ------------------------------
@torch.no_grad()
def get_whittle_table(actor: nn.Module, env: lineEnv, device):
    env_emb = torch.tensor([env.p, env.q, float(env.OptX)], dtype=torch.float32, device=device).unsqueeze(0)
    return [float(actor(torch.tensor([s], dtype=torch.float32, device=device).unsqueeze(0), env_emb).item()) for s in range(env.N)]

# ------------------------------
# MAML 训练主程序 (FOMAML)
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob_file", default="prob_pairs.json")
    ap.add_argument("--num", type=int, default=10)
    # 环境参数
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--OptX", type=int, default=99)
    # 元训练参数
    ap.add_argument("--meta_iterations", type=int, default=4000)
    ap.add_argument("--tasks_per_meta_batch", type=int, default=5)
    ap.add_argument("--adapt_steps", type=int, default=3)
    ap.add_argument("--adapt_lr", type=float, default=1e-4)
    ap.add_argument("--meta_lr", type=float, default=1e-4)
    # 其他 RL 超参
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--warmup_ratio", type=float, default=0.2)
    ap.add_argument("--epsilon", type=float, default=0.05)
    # λ 范围
    ap.add_argument("--lam_low", type=float, default=-2.0)
    ap.add_argument("--lam_high", type=float, default=2.0)
    # 其他
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_root", default=None)
    args = ap.parse_args()

    # 固定随机种子
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 载入 (p,q) 组合
    prob_pairs = json.load(open(args.prob_file, "r", encoding="utf-8"))[:args.num]
    if len(prob_pairs) < args.num:
        raise ValueError("概率对数量不足")

    save_root = args.save_root or f"maml_reinforce_{args.num}"; os.makedirs(save_root, exist_ok=True)

    # 预先采样 λ 序列（所有臂共用）
    total_episodes = args.meta_iterations * args.adapt_steps
    lam_seq = [random.uniform(args.lam_low, args.lam_high) for _ in range(total_episodes)]
    lam_ptr = 0

    # 构建所有 Task 环境对象
    tasks = []
    for idx, pq in enumerate(prob_pairs):
        env = lineEnv(seed=args.seed, N=args.N, OptX=args.OptX, p=float(pq["p"]), q=float(pq["q"]))
        tasks.append(env)

    # 初始化 meta‑actor
    meta_actor = Actor(state_dim=1).to(device)
    meta_optimizer = optim.Adam(meta_actor.parameters(), lr=args.meta_lr)

    # 记录损失
    meta_loss_hist = []

    # ---------- 元训练循环 ----------
    for it in trange(args.meta_iterations, desc="Meta‑train"):
        meta_optimizer.zero_grad()
        # 随机采样 tasks_per_meta_batch 个任务
        task_indices = random.sample(range(args.num), args.tasks_per_meta_batch)
        meta_batch_loss = 0.0
        for ti in task_indices:
            env = tasks[ti]
            # 1) Clone meta‑actor
            actor_clone = copy.deepcopy(meta_actor)
            clone_optim = optim.SGD(actor_clone.parameters(), lr=args.adapt_lr)

            # 2) Inner‑loop adaptation K 步
            for k in range(args.adapt_steps):
                batch_states = random.sample(list(range(env.N)), args.batch_size)
                lam_value = lam_seq[lam_ptr % len(lam_seq)]; lam_ptr += 1
                force_random = (k == 0)  # 第一步完全随机探索
                loss = reinforce_loss(
                    actor_clone, env, device, lam_value, args.gamma, args.max_steps,
                    batch_states, epsilon=args.epsilon, force_random=force_random,
                )
                clone_optim.zero_grad(); loss.backward(); clone_optim.step()

            # 3) Meta‑loss on fresh episodes (first‑order, no 2nd‑grad)
            eval_states = random.sample(list(range(env.N)), args.batch_size)
            lam_value = lam_seq[lam_ptr % len(lam_seq)]; lam_ptr += 1
            meta_loss = reinforce_loss(
                actor_clone, env, device, lam_value, args.gamma, args.max_steps,
                eval_states, epsilon=0.0, force_random=False,
            )
            meta_batch_loss += meta_loss.item()

            # 4) Compute grads w.r.t actor_clone params (first‑order)
            grads = torch.autograd.grad(meta_loss, actor_clone.parameters())
            # Accumulate onto meta‑params
            for p_meta, g in zip(meta_actor.parameters(), grads):
                if p_meta.grad is None:
                    p_meta.grad = g.clone().detach()
                else:
                    p_meta.grad += g.clone().detach()

        # 平均梯度
        for p_meta in meta_actor.parameters():
            p_meta.grad /= args.tasks_per_meta_batch
        meta_optimizer.step()
        meta_loss_hist.append(meta_batch_loss / args.tasks_per_meta_batch)

    # 保存 meta 模型 & 损失曲线
    torch.save(meta_actor.state_dict(), os.path.join(save_root, "meta_actor.pt"))
    plt.figure(); plt.plot(meta_loss_hist); plt.xlabel("Meta‑iteration"); plt.ylabel("Meta‑batch loss")
    plt.tight_layout(); plt.savefig(os.path.join(save_root, "meta_loss.png"), dpi=200); plt.close()

    # ---------- 评估：输出每个训练臂的 Whittle 曲线 ----------
    all_tables = []
    for idx, env in enumerate(tasks):
        tbl = get_whittle_table(meta_actor, env, device)
        all_tables.append({"arm": idx, "p": env.p, "q": env.q, "whittle": tbl})
        json.dump({"p": env.p, "q": env.q, "whittle": tbl}, open(os.path.join(save_root, f"arm{idx:02d}_whittle.json"), "w"), indent=2)

    json.dump(all_tables, open(os.path.join(save_root, "all_whittle_tables.json"), "w"), indent=2)

    # 画所有曲线
    plt.figure(figsize=(6,4))
    for d in all_tables:
        plt.plot(d["whittle"], label=f"Arm{d['arm']} p={d['p']:.2f} q={d['q']:.2f}")
    plt.xlabel("State s"); plt.ylabel("λ̂(s)")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(save_root, "whittle_tables.png"), dpi=250)
    plt.close()

    print("✓ Meta‑training complete. Results saved in", save_root)

if __name__ == "__main__":
    main()
