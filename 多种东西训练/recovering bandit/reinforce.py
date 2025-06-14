from __future__ import annotations

import os, json, argparse, random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import trange

from rc_env import recoveringBanditsEnv  # (ensure rc_env.py is in PYTHONPATH)


#尝试一下强制单增来计算结果
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(*size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int = 3, hidden_dim: int = 256, out_init_w: float = 3e-3):
        super().__init__()
        self.input_dim = state_dim + embed_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)  # λ̂(s) logit
        self.act = nn.ReLU()
        self.init_weights(out_init_w)

    def init_weights(self, out_init_w):
        # 隐层用 fanin_init
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()
        self.fc3.bias.data.zero_()
        # 输出层权重小区间
        self.fc_out.weight.data.uniform_(-out_init_w, out_init_w)
        self.fc_out.bias.data.zero_()

    def forward(self, state: torch.Tensor, env_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, env_emb], dim=-1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.fc_out(x)

"""class Actor(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int = 3, hidden_dim: int = 256):
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
        return self.fc_out(x)"""

# ---------------------------------------------------------
# 单批 REINFORCE 更新，新增 temperature
# ---------------------------------------------------------
def reinforce_update(
    *,
    actor: nn.Module,
    opt: torch.optim.Optimizer,
    env: recoveringBanditsEnv,
    gamma: float,
    device: torch.device,
    lam_value: float,
    max_steps: int,
    batch_states: List[int],
    temperature: float = 1.0,   # <--- 新增
    epsilon: float = 0.0,
    force_random: bool = False,
) -> float:
    batch_losses = []
    env_emb_const = torch.tensor(
        [env.theta0, env.theta1, env.theta2], dtype=torch.float32, device=device
    ).unsqueeze(0)

    for start_state in batch_states:
        log_probs, rewards = [], []
        env.reset()
        env.arm[0] = start_state  # directly set current wait time

        for _ in range(max_steps):
            s = (
                torch.tensor([env.arm[0]], dtype=torch.float32, device=device)
                .unsqueeze(0)
            )  # (1,1)

            # π(a=1|s) via sigmoid((logit − λ)/temperature)
            logit = actor(s, env_emb_const)
            p1 = torch.sigmoid((logit - lam_value) / temperature)    # <--- 改动
            probs = torch.cat([1 - p1, p1], dim=-1).squeeze(0)  # (2,)
            dist = torch.distributions.Categorical(probs=probs)

            if force_random or random.random() < epsilon:
                a = torch.tensor(random.randint(0, 1), device=device)
            else:
                a = dist.sample()

            log_probs.append(dist.log_prob(a))
            _, r, _, _ = env.step(a.item())
            if a.item() == 1:
                r -= lam_value  # pulling cost
            rewards.append(r)

        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # —— normalise for variance reduction —— 
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantage = returns

        entropy = torch.stack([dist.entropy()] * len(returns)).mean()
        traj_loss = -torch.sum(torch.stack(log_probs) * advantage)
        traj_loss = traj_loss - 0.01 * entropy
        batch_losses.append(traj_loss)

    loss = torch.stack(batch_losses).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    opt.step()

    return loss.item()

# ---------------------------------------------------------
# Whittle table estimation (no grad)
# ---------------------------------------------------------
@torch.no_grad()
def get_whittle_table(actor: nn.Module, env: recoveringBanditsEnv, device) -> List[float]:
    env_emb = torch.tensor(
        [env.theta0, env.theta1, env.theta2], dtype=torch.float32, device=device
    ).unsqueeze(0)
    tbl = []
    for s in range(1, env.maxWait + 1):
        state_t = torch.tensor([s], dtype=torch.float32, device=device).unsqueeze(0)
        tbl.append(float(actor(state_t, env_emb).item()))
    return tbl

# ---------------------------------------------------------
#                  M A I N   P R O G R A M
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # —— experiment setup ——
    ap.add_argument("--theta_file", default="theta_pairs_recovering.json")
    ap.add_argument("--num", type=int, default=10, help="number of arms to train")
    ap.add_argument("--save_root", default=None)

    # —— RL hyper‑params ——
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)

    # —— environment ——
    ap.add_argument("--noise_var", type=float, default=0.05)
    ap.add_argument("--max_wait", type=int, default=None, help="override maxWait in file")

    # —— exploration ——
    ap.add_argument("--warmup_ratio", type=float, default=0.2)
    ap.add_argument("--epsilon", type=float, default=0.05)

    # —— λ range ——
    ap.add_argument("--lam_low", type=float, default=-5.0)
    ap.add_argument("--lam_high", type=float, default=10.0)

    # —— temperature (NEW) ——
    ap.add_argument("--temperature", type=float, default=0.1, help="sigmoid temperature")   # <--- 新增

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("➤ Device:", device)

    raw = json.load(open(args.theta_file, "r", encoding="utf-8"))
    theta_pairs = raw["theta_pairs"][: args.num]
    if len(theta_pairs) < args.num:
        raise ValueError("θ‑pairs fewer than requested num")

    max_wait = args.max_wait or raw.get("maxWait", 20)
    save_root = args.save_root or f"reinforce_recovering_{args.num}"
    os.makedirs(save_root, exist_ok=True)
    lam_seq = [random.uniform(args.lam_low, args.lam_high) for _ in range(args.episodes)]
    warmup_episodes = int(args.episodes * args.warmup_ratio)

    all_tables = []
    for idx, thetas in enumerate(theta_pairs):
        theta0, theta1, theta2 = map(float, thetas)
        env = recoveringBanditsEnv(
            seed=args.seed,
            thetaVals=(theta0, theta1, theta2),
            noiseVar=args.noise_var,
            maxWait=max_wait,
        )

        actor = Actor(state_dim=1).to(device)
        opt = optim.Adam(actor.parameters(), lr=args.lr)

        loss_hist = []
        states_cycle = list(range(1, max_wait + 1))
        cycle_ptr = 0

        desc = f"Arm{idx}-θ0{theta0:.2f}θ1{theta1:.2f}"
        for ep in trange(args.episodes, desc=desc, leave=False):
            if cycle_ptr == 0:
                random.shuffle(states_cycle)
            batch_states = [states_cycle[(cycle_ptr + i) % max_wait] for i in range(args.batch_size)]
            cycle_ptr = (cycle_ptr + args.batch_size) % max_wait

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
                temperature=args.temperature,  # <--- 新增传参
                epsilon=eps_val,
                force_random=force_random,
            )
            loss_hist.append(loss)

        # —— save artefacts ——
        model_path = os.path.join(
            save_root, f"arm{idx:02d}_t0{theta0:.3f}_t1{theta1:.2f}.pt"
        )
        torch.save(actor.state_dict(), model_path)

        plt.figure(figsize=(6, 3))
        plt.plot(loss_hist, label="REINFORCE loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title(f"Arm {idx} (θ0={theta0:.2f}, θ1={theta1:.2f})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_root, f"arm{idx:02d}_loss.png"), dpi=200)
        plt.close()

        tbl = get_whittle_table(actor, env, device)
        w_json_path = os.path.join(save_root, f"arm{idx:02d}_whittle.json")
        json.dump({"theta0": theta0, "theta1": theta1, "theta2": theta2, "whittle": tbl}, open(w_json_path, "w"), indent=2)

        all_tables.append({"arm": idx, "theta0": theta0, "theta1": theta1, "theta2": theta2, "whittle": tbl})
        print(f"[Arm {idx}] done — saved model → {model_path}")

    json.dump(all_tables, open(os.path.join(save_root, "all_whittle_tables.json"), "w"), indent=2)
    print("✓ Training complete. Results stored in", save_root)


if __name__ == "__main__":
    main()
