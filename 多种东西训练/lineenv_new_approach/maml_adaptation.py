
import os, json, random, argparse, copy
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from lineEnv import lineEnv            # 你的 RMAB 单臂环境

# -------------------- Actor 网络 --------------------
class Actor(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(state_dim + 3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)             # 输出 Whittle logit λ̂(s)
        )

    def forward(self, state, env_emb):
        # state:(B,1)  env_emb:(B,3)
        x = torch.cat([state, env_emb], dim=-1)
        return self.net(x)                   # (B,1)

# ------------------ REINFORCE 损失 ------------------
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
            s = torch.tensor([[env.state]], dtype=torch.float32, device=device)       # (1,1)
            env_emb = torch.tensor([[env.p, env.q, float(env.OptX)]],
                                   dtype=torch.float32, device=device)               # (1,3)
            logit = actor(s, env_emb)            # λ̂(s)
            p1 = torch.sigmoid(logit - lam_value)
            probs = torch.cat([1 - p1, p1], dim=-1).squeeze(0)  # (2,)
            dist = torch.distributions.Categorical(probs=probs)

            if force_random or random.random() < epsilon:
                a = torch.tensor(random.randint(0, 1), device=device)
            else:
                a = dist.sample()

            log_probs.append(dist.log_prob(a))
            _, r, _, _ = env.step(a.item())
            if a.item() == 1:
                r -= lam_value
            rewards.append(r)

        # 折扣回报
        R = 0.0; returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 归一化
        traj_loss = -torch.sum(torch.stack(log_probs) * returns)
        losses.append(traj_loss)
    return torch.stack(losses).mean()

# ------------- 生成整张 Whittle 指数表 --------------
@torch.no_grad()
def get_whittle_table(actor: nn.Module, env: lineEnv, device):
    env_emb = torch.tensor([[env.p, env.q, float(env.OptX)]],
                           dtype=torch.float32, device=device)
    values = []
    for s in range(env.N):
        s_t = torch.tensor([[s]], dtype=torch.float32, device=device)
        values.append(float(actor(s_t, env_emb).item()))
    return values

# ====================== 主程序 ======================
def main():
    ap = argparse.ArgumentParser()
    # ------------- 推断权重相关 -------------
    ap.add_argument("--meta_model", default=None,
                    help="训练好的 meta-actor 权重 (.pt)。若为空则自动推断")
    ap.add_argument("--num", type=int, default=10,
                    help="训练时使用的 tasks 数，用于推断默认权重目录")
    # ------------- 基本文件 -------------
    ap.add_argument("--prob_file", default="prob_pairs.json")
    ap.add_argument("--save_root", default="maml_adaptation_results")
    # ------------- 环境参数 -------------
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--OptX", type=int, default=99)
    # ------------- 自适应参数 -------------
    ap.add_argument("--adapt_steps", nargs="+", type=int, default=[0,1, 5, 10, 50],
                    help="需要输出结果的内循环步数列表")
    ap.add_argument("--adapt_lr", type=float, default=1e-4)
    # ------------- RL 超参 -------------
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epsilon", type=float, default=0.05)
    # ------------- λ 采样区间 -------------
    ap.add_argument("--lam_low", type=float, default=-2.0)
    ap.add_argument("--lam_high", type=float, default=2.0)
    # ------------- 其余 -------------
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    # ---------- 自动推断 meta_actor.pt ----------
    if args.meta_model is None:
        guess_root = f"maml_reinforce_{args.num}"
        args.meta_model = os.path.join(guess_root, "meta_actor.pt")
        if not os.path.isfile(args.meta_model):
            raise FileNotFoundError(
                f"自动推断权重失败：{args.meta_model} 不存在。\n"
                "请检查 --num 是否正确或显式传入 --meta_model 路径。")
    print("✓ Using meta model:", args.meta_model)

    # ---------- 随机种子 ----------
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✓ Device:", device)

    # ---------- 载入 (p,q) 组合 ----------
    prob_pairs = json.load(open(args.prob_file, "r", encoding="utf-8"))
    num_envs = len(prob_pairs)
    print(f"✓ Loaded {num_envs} arms from {args.prob_file}")

    # ---------- 创建结果目录 ----------
    os.makedirs(args.save_root, exist_ok=True)

    # ---------- 加载 meta-actor ----------
    meta_actor = Actor(state_dim=1).to(device)
    meta_actor.load_state_dict(torch.load(args.meta_model, map_location=device))
    meta_actor.eval()                              # meta-actor 本身不更新

    # ========== 对每个适应步数 M 循环 ==========
    for M in args.adapt_steps:
        subdir = os.path.join(args.save_root, f"adaptation{M}")
        os.makedirs(subdir, exist_ok=True)
        print(f"\n=== Adaptation with {M} step(s) ===")

        all_tables = []

        # ------- 遍历全部 Arms -------
        for idx, pq in enumerate(prob_pairs):
            env = lineEnv(seed=args.seed, N=args.N, OptX=args.OptX,
                          p=float(pq["p"]), q=float(pq["q"]))

            # 复制 meta-actor → actor_i
            actor_i = copy.deepcopy(meta_actor).to(device).train()
            optim_i = optim.SGD(actor_i.parameters(), lr=args.adapt_lr)

            # 预采样 λ 序列（每一步一个）
            lam_seq = [random.uniform(args.lam_low, args.lam_high) for _ in range(M)]

            # ------- 内循环自适应 -------
            for step in range(M):
                batch_states = random.sample(range(env.N), args.batch_size)
                lam_value = lam_seq[step]
                loss = reinforce_loss(
                    actor_i, env, device, lam_value,
                    args.gamma, args.max_steps, batch_states,
                    epsilon=args.epsilon, force_random=(step == 0)
                )
                optim_i.zero_grad(); loss.backward(); optim_i.step()

            # ------- 生成并保存 Whittle 表 -------
            tbl = get_whittle_table(actor_i.eval(), env, device)
            arm_json = {
                "arm": idx,
                "p": env.p,
                "q": env.q,
                "whittle": tbl,
            }
            all_tables.append(arm_json)

            json.dump(
                arm_json,
                open(os.path.join(subdir, f"arm{idx:03d}_whittle.json"), "w"),
                indent=2,
            )

        # ------- 保存汇总 -------
        json.dump(all_tables,
                  open(os.path.join(subdir, "all_whittle_tables.json"), "w"),
                  indent=2)
        print(f"✓ Results saved in {subdir}")

    print("\n✔ All adaptation finished. Root dir:", args.save_root)

# ------------------------------ 主入口 ------------------------------
if __name__ == "__main__":
    main()
