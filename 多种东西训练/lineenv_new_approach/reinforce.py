# ---------------------------------------------------------
# reinforce_trad_batch.py  (pre‑sampled λ sequence per episode)
# ---------------------------------------------------------
import os, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import trange

from lineEnv import lineEnv           # ↖️ 你的 RMAB 环境

# ----------------- Actor 网络 -----------------
class Actor(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.input_dim = state_dim + 3          # state(1) + (p,q,optX)
        hidden_dim = 256
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)     # 输出单标量 logit = λ̂(s)
        self.activation = nn.ReLU()

    def forward(self, state, env_emb):
        """state:(B,1), env_emb:(B,3) -> (B,1)"""
        x = torch.cat([state, env_emb], dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return self.fc5(x)

# ----------------- 单批 REINFORCE 更新 -----------------

def reinforce_update(
    *,
    actor: nn.Module,
    opt: torch.optim.Optimizer,
    env: lineEnv,
    gamma: float,
    device: torch.device,
    lam_value: float,
    max_steps: int,
    batch_states,
    epsilon: float = 0.0,
    force_random: bool = False,
):
    """对给定 λ 执行一次批量 REINFORCE。
    lam_value 在调用层已固定，保证所有臂同顺序使用同一 λ 序列。
    """
    batch_losses = []

    for start_state in batch_states:
        log_probs, rewards = [], []
        env.reset(); env.state = start_state

        for _ in range(max_steps):
            s = torch.tensor([env.state], dtype=torch.float32, device=device).unsqueeze(0)
            env_emb = torch.tensor([env.p, env.q, float(env.OptX)], dtype=torch.float32, device=device).unsqueeze(0)

            # π(a=1|s) 由 sigmoid(logit−λ)
            logit = actor(s, env_emb)
            p1 = torch.sigmoid(logit - lam_value)
            probs = torch.cat([1 - p1, p1], dim=-1).squeeze(0)
            dist = torch.distributions.Categorical(probs=probs)

            # ε‑greedy or full‑random
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
        """R = 0.0; returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        traj_loss = -torch.sum(torch.stack(log_probs) * returns)"""
        # 折扣回报
        R = 0.0; returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # ⭐ normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ⭐ optional baseline
        advantage = returns  # 或者 returns - returns.mean()

        # ⭐ entropy optional
        dist = torch.distributions.Categorical(probs=probs)
        entropy = torch.stack([d.entropy() for d in [dist] * len(returns)]).mean()

        # Loss
        traj_loss = -torch.sum(torch.stack(log_probs) * advantage)
        traj_loss = traj_loss - 0.01 * entropy

        batch_losses.append(traj_loss)

    loss = torch.stack(batch_losses).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    opt.step()
    return loss.item()

# ----------------- 估计整张 Whittle 表 -----------------
@torch.no_grad()
def get_whittle_table(actor, env, device):
    env_emb = torch.tensor([env.p, env.q, float(env.OptX)], dtype=torch.float32, device=device).unsqueeze(0)
    return [float(actor(torch.tensor([s], dtype=torch.float32, device=device).unsqueeze(0), env_emb).item()) for s in range(env.N)]

# ----------------- 主程序 -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob_file", default="prob_pairs.json")
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--save_root", default=None)
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    # 环境
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--OptX", type=int, default=99)
    # 探索
    ap.add_argument("--warmup_ratio", type=float, default=0.2)
    ap.add_argument("--epsilon", type=float, default=0.05)
    # λ 范围
    ap.add_argument("--lam_low",  type=float, default=-2.0)
    ap.add_argument("--lam_high", type=float, default= 2.0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    prob_pairs = json.load(open(args.prob_file, "r", encoding="utf-8"))[:args.num]
    if len(prob_pairs) < args.num:
        raise ValueError("概率对数量不足")

    save_root = args.save_root or f"reinforce_trad_{args.num}"; os.makedirs(save_root, exist_ok=True)

    # 预先采样 λ 序列（同一序列供所有臂使用）
    lam_seq = [random.uniform(args.lam_low, args.lam_high) for _ in range(args.episodes)]
    warmup_episodes = int(args.episodes * args.warmup_ratio)

    all_tables = []
    for idx, pq in enumerate(prob_pairs):
        p, q = float(pq["p"]), float(pq["q"])
        env = lineEnv(seed=args.seed, N=args.N, OptX=args.OptX, p=p, q=q)
        actor = Actor(state_dim=1).to(device)
        opt   = optim.Adam(actor.parameters(), lr=args.lr)

        loss_hist, states_cycle = [], list(range(env.N)); cycle_ptr = 0
        for ep in trange(args.episodes, desc=f"Arm{idx}-p{p:.2f}q{q:.2f}", leave=False):
            if cycle_ptr == 0:
                random.shuffle(states_cycle)
            batch_states = [states_cycle[(cycle_ptr+i)%env.N] for i in range(args.batch_size)]
            cycle_ptr = (cycle_ptr + args.batch_size) % env.N

            force_random = ep < warmup_episodes
            eps_val = 0.0 if force_random else args.epsilon
            lam_val = lam_seq[ep]

            loss = reinforce_update(
                actor=actor,
                opt=opt,
                env=env,
                gamma=args.gamma,
                device=device,
                lam_value=lam_val,
                max_steps=args.max_steps,
                batch_states=batch_states,
                epsilon=eps_val,
                force_random=force_random,
            )
            loss_hist.append(loss)

        # 保存模型与曲线
        w_path = os.path.join(save_root, f"arm{idx:02d}_p{p:.3f}_q{q:.3f}.pt")
        torch.save(actor.state_dict(), w_path)
        plt.figure(); plt.plot(loss_hist); plt.xlabel("Episode"); plt.ylabel("REINFORCE Loss")
        plt.title(f"Arm {idx} (p={p:.2f}, q={q:.2f})"); plt.tight_layout()
        plt.savefig(os.path.join(save_root, f"arm{idx:02d}_loss.png"), dpi=200); plt.close()

        tbl = get_whittle_table(actor, env, device)
        json.dump({"p": p, "q": q, "whittle": tbl}, open(os.path.join(save_root, f"arm{idx:02d}_whittle.json"), "w"), indent=2)
        all_tables.append({"arm": idx, "p": p, "q": q, "whittle": tbl})
        print(f"[Arm {idx}] done — saved to {w_path}")

    json.dump(all_tables, open(os.path.join(save_root, "all_whittle_tables.json"), "w"), indent=2)
    print("✓ 全部完成，结果存于", save_root)

if __name__ == "__main__":
    main()
