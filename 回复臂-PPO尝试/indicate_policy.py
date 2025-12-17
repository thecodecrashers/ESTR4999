import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

from rc_env import recoveringBanditsEnv
from model import ActorCritic

# ================= 配置 =================
device = torch.device("cpu")

GAMMA = 0.99
GAE_LAMBDA = 0.95
TOTAL_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 50

LAMBDA_MIN, LAMBDA_MAX = 0, 11
NUM_LAMBDAS = 10

ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
GRAD_CLIP_NORM = 5.0


EPSILON = 0.1

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


def train():
    env = recoveringBanditsEnv(
        seed=42,
        thetaVals=[10, 0.5, 0.0],
        noiseVar=0,
        maxWait=50
    )

    ac = ActorCritic(state_dim=1).to(device)

    actor_opt  = optim.Adam(ac.actor.parameters(),  lr=ACTOR_LR)
    critic_opt = optim.Adam(ac.critic.parameters(), lr=CRITIC_LR)

    lambda_bins = np.linspace(LAMBDA_MIN, LAMBDA_MAX, NUM_LAMBDAS + 1)

    CRITIC_BATCH_SIZE = 8
    ACTOR_BATCH_SIZE  = 32
    RANGE_COEF = 1e-2

    tqdm_bar = tqdm(range(TOTAL_EPISODES), desc="training")

    # ===== buffers =====
    c_s_buf, c_lam_buf, c_tgt_buf = [], [], []
    a_s_buf, a_lam_buf, a_act_buf, a_td_buf = [], [], [], []

    for ep in tqdm_bar:
        cl_sum, cl_cnt = 0.0, 0
        al_sum, al_cnt = 0.0, 0

        for lam_idx in range(NUM_LAMBDAS):
            lam = np.random.uniform(lambda_bins[lam_idx], lambda_bins[lam_idx + 1])
            lam_t = torch.tensor([[lam]], dtype=torch.float32, device=device)

            obs = env.reset()
            done = False
            step = 0

            while not done and step < MAX_STEPS_PER_EPISODE and obs[0] < env.maxWait:
                step += 1
                s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    z = ac.act(s).item()
                    greedy_a = 1 if z > lam else 0

                if np.random.rand() < EPSILON:
                    a = np.random.choice([0, 1])
                else:
                    a = greedy_a

                r = env._calReward(a, obs[0]) - (lam if a == 1 else 0.0)
                next_obs, _, done, _ = env.step(a)
                next_obs = np.array([env.arm[0]], dtype=np.float32)
                s_next = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    v_next = ac.value(s_next, lam_t)
                    td_target = r + GAMMA * v_next * (1 - float(done))

                # ===== store critic sample =====
                c_s_buf.append(s)
                c_lam_buf.append(lam_t)
                c_tgt_buf.append(td_target.detach())

                # ===== TD error =====
                v = ac.value(s, lam_t)
                delta = (td_target - v).detach()

                # ===== store actor sample =====
                a_s_buf.append(s)
                a_lam_buf.append(lam_t)
                a_act_buf.append(a)
                a_td_buf.append(delta)

                # ===== critic batch update =====
                if len(c_s_buf) >= CRITIC_BATCH_SIZE:
                    S = torch.cat(c_s_buf)
                    L = torch.cat(c_lam_buf)
                    T = torch.cat(c_tgt_buf)

                    loss = (ac.value(S, L) - T).pow(2).mean()
                    critic_opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(ac.critic.parameters(), GRAD_CLIP_NORM)
                    critic_opt.step()

                    cl_sum += loss.item()
                    cl_cnt += 1
                    c_s_buf.clear(); c_lam_buf.clear(); c_tgt_buf.clear()

                # ===== actor batch update =====
                if len(a_s_buf) >= ACTOR_BATCH_SIZE:
                    S = torch.cat(a_s_buf)
                    L = torch.cat(a_lam_buf)
                    Z = ac.act(S)

                    A = torch.tensor(a_act_buf, device=device).unsqueeze(1)
                    D = torch.cat(a_td_buf)

                    raise_mask = (A == 0) & (D > 0)
                    lower_mask = (A == 1) & (D < 0)

                    dir_loss = (
                        raise_mask * torch.abs(D) * (L - Z) +
                        lower_mask * torch.abs(D) * (Z - L)
                    )

                    range_loss = (
                        torch.relu(Z - LAMBDA_MAX).pow(2) +
                        torch.relu(LAMBDA_MIN - Z).pow(2)
                    )

                    loss = dir_loss.mean() + RANGE_COEF * range_loss.mean()

                    actor_opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(ac.actor.parameters(), GRAD_CLIP_NORM)
                    actor_opt.step()

                    al_sum += loss.item()
                    al_cnt += 1
                    a_s_buf.clear(); a_lam_buf.clear()
                    a_act_buf.clear(); a_td_buf.clear()

                obs = next_obs

        tqdm_bar.set_description(
            f"Ep {ep+1} | C_loss {cl_sum/max(1,cl_cnt):.4f} | "
            f"A_loss {al_sum/max(1,al_cnt):.4f}"
        )

    torch.save(
        {"actor": ac.actor.state_dict(), "critic": ac.critic.state_dict()},
        os.path.join(SAVE_DIR, "index_td_batch.pt")
    )

    return ac


# ================= 可视化 =================
def visualize_index(ac, max_state=40):
    ac.actor.eval()
    states = torch.arange(1, max_state + 1, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        index_preds = ac.actor(states)

    plt.figure(figsize=(8, 5))
    plt.plot(states.cpu().numpy(), index_preds.cpu().numpy(), label="Learned Index")
    plt.xlabel("State")
    plt.ylabel("Index Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ================= 主入口 =================
if __name__ == "__main__":
    ac_model = train()
    visualize_index(ac_model)
