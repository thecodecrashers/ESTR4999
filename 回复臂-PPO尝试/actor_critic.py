import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rc_env import recoveringBanditsEnv
from model import ActorCritic  # act(state)->index z (B,1), value(state, lambda)->V (B,1)

# ========== Config ==========
device = torch.device("cpu")  # or cuda

GAMMA = 0.99
TOTAL_EPISODES = 100000
MAX_STEPS_PER_EPISODE = 50

LAMBDA_MIN, LAMBDA_MAX = 0, 11
NUM_LAMBDAS = 10

ACTOR_LR = 1e-2
CRITIC_LR = 1e-3
N_CRITIC_UPDATES = 16
GRAD_CLIP_NORM = 5.0

EPSILON = 0.0

# ===== entropy + soft range regularization =====
ENTROPY_COEF = 0.01
RANGE_COEF   = 1e-2


def train():
    env = recoveringBanditsEnv(seed=42, thetaVals=[10, 0.5, 0.0], noiseVar=0, maxWait=50)
    ac = ActorCritic(state_dim=1).to(device)
    state = env.reset()
    actor_opt  = optim.Adam(ac.actor.parameters(),  lr=ACTOR_LR)
    critic_opt = optim.Adam(ac.critic.parameters(), lr=CRITIC_LR)

    lambda_bins = np.linspace(LAMBDA_MIN, LAMBDA_MAX, NUM_LAMBDAS + 1)
    tqdm_bar = tqdm(range(TOTAL_EPISODES), desc="training")

    # ===== batch config =====
    ACTOR_BATCH_SIZE  = 64

    ENTROPY_COEF = 0.01
    RANGE_COEF   = 1e-2

    for ep in tqdm_bar:
        # ===== buffers =====
        s_buf, lam_buf, a_buf = [], [], []
        r_buf, done_buf = [], []
        logp_buf, ent_buf = [], []

        actor_losses, critic_losses = [], []
        max=0
        for lam_idx in range(NUM_LAMBDAS):
            lam = np.random.uniform(lambda_bins[lam_idx], lambda_bins[lam_idx + 1])
            lam_t = torch.tensor([[lam]], dtype=torch.float32, device=device)

#            state = env.reset()
            env.arm[0] = np.random.randint(1, env.maxWait + 1)
            state = np.array([env.arm[0]], dtype=np.float32)
            done = False
            step = 0

            #while not done and step < MAX_STEPS_PER_EPISODE and state[0] < env.maxWait:
            while not done and step < MAX_STEPS_PER_EPISODE:
                # if state.item()>max:
                #     print(state.item())
                #     max=state.item()
                step += 1
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                # ===== Actor: stochastic policy =====
                z = ac.act(s)                          # index
                logits = z - lam_t
                prob_1 = torch.sigmoid(logits)

                dist = torch.distributions.Bernoulli(prob_1)
                a_sample = dist.sample()
                logp = dist.log_prob(a_sample)
                entropy = dist.entropy()

                # ===== ε-greedy exploration =====
                if np.random.rand() < EPSILON:
                    a = np.random.choice([0, 1])
                    logp = torch.zeros_like(logp)  # 不从 policy 学
                    entropy = torch.zeros_like(entropy)
                else:
                    a = int(a_sample.item())
                # ===== env step =====
                r = env._calReward(a, state[0]) - lam * a
                next_state, _, done, _ = env.step(a)
                next_state = np.array([env.arm[0]], dtype=np.float32)

                # ===== store transition =====
                s_buf.append(s)
                lam_buf.append(lam_t)
                a_buf.append(a)
                r_buf.append(r)
                done_buf.append(done)
                logp_buf.append(logp)
                ent_buf.append(entropy)

                state = next_state

                # ===== batch update =====
                if len(s_buf) >= ACTOR_BATCH_SIZE:
                    # --- build tensors ---
                    S   = torch.cat(s_buf)
                    LAM = torch.cat(lam_buf)
                    A   = torch.tensor(a_buf, device=device).unsqueeze(1)
                    R   = torch.tensor(r_buf, device=device).unsqueeze(1)
                    D   = torch.tensor(done_buf, device=device).unsqueeze(1).float()

                    LOGP = torch.cat(logp_buf)
                    ENT  = torch.cat(ent_buf)

                    # --- critic TD ---
                    V = ac.value(S, LAM)
                    with torch.no_grad():
                        S_next = torch.cat(
                            [torch.tensor([[env.arm[0]]], dtype=torch.float32, device=device)]
                            * len(S)
                        )
                        V_next = ac.value(S_next, LAM)
                        td_target = R + GAMMA * V_next * (1 - D)

                    delta = td_target - V

                    # ===== critic update =====
                    critic_loss = delta.pow(2).mean()
                    critic_opt.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(ac.critic.parameters(), GRAD_CLIP_NORM)
                    critic_opt.step()

                    # ===== actor loss =====
                    pg_loss = -(LOGP * delta.detach()).mean()
                    entropy_loss = -ENT.mean()

                    # --- soft range constraint on index ---
                    z_all = ac.act(S)
                    range_loss = (
                        torch.relu(z_all - LAMBDA_MAX).pow(2) +
                        torch.relu(LAMBDA_MIN - z_all).pow(2)
                    ).mean()

                    actor_loss = (
                        pg_loss
                        + ENTROPY_COEF * entropy_loss
                        + RANGE_COEF * range_loss
                    )

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(ac.actor.parameters(), GRAD_CLIP_NORM)
                    actor_opt.step()

                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())

                    # clear buffers
                    s_buf.clear(); lam_buf.clear(); a_buf.clear()
                    r_buf.clear(); done_buf.clear()
                    logp_buf.clear(); ent_buf.clear()

        tqdm_bar.set_description(
            f"Ep {ep+1} | "
            f"A_loss {np.mean(actor_losses) if actor_losses else 0:.4f} | "
            f"C_loss {np.mean(critic_losses) if critic_losses else 0:.4f}"
        )

    return ac


def visualize_index(ac, max_state=40):
    ac.actor.eval()
    states = torch.arange(1, max_state + 1, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        index_preds = ac.actor(states)
    plt.figure(figsize=(8, 5))
    plt.plot(states.cpu().numpy(), index_preds.cpu().numpy(), label="Whittle Index")
    plt.xlabel("State")
    plt.ylabel("Predicted Index")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ac_model = train()
    visualize_index(ac_model)




#使用BCEloss来训练Actor网络，然后使用critic的指示来强制指导actor是可以训练出来的
#region
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# from rc_env import recoveringBanditsEnv   # 你的环境
# from model  import ActorCritic            # 你的网络 (需定义 act(), value())

# # ========= 设备选择 =========
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========= 超参数 =========
# GAMMA                = 0.99
# GAE_LAMBDA           = 0.95
# TOTAL_EPISODES       = 10000
# MAX_STEPS_PER_EPISODE= 50

# LAMBDA_MIN, LAMBDA_MAX = 1, 11
# NUM_LAMBDAS            = 10

# ACTOR_LR  = 1e-4
# CRITIC_LR = 1e-5
# N_CRITIC_UPDATES = 10
# ENTROPY_COEF    = 0.0
# GRAD_CLIP_NORM  = 5.0

# def visualize_index(ac, max_state=40):
#     ac.actor.eval()
#     states = torch.arange(1, max_state + 1, dtype=torch.float32).unsqueeze(1).to(device)
#     with torch.no_grad():
#         index_preds = ac.actor(states)
#     plt.figure(figsize=(8, 5))
#     plt.plot(states.cpu().numpy().squeeze(),
#              index_preds.cpu().numpy().squeeze(),
#              label="Whittle Index Estimate", marker='o')
#     plt.xlabel("State")
#     plt.ylabel("Predicted Index")
#     plt.title("Actor Network: Whittle Index vs. State")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def compute_gae(rewards, values, next_value, dones):
#     values = values + [next_value]
#     gae, returns = 0.0, []
#     for t in reversed(range(len(rewards))):
#         delta = rewards[t] + GAMMA * values[t+1] * (1 - dones[t]) - values[t]
#         gae   = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
#         returns.insert(0, gae + values[t])
#     return returns

# def train():
#     env = recoveringBanditsEnv(
#         seed       = 42,
#         thetaVals  = [10, 0.5, 0.0],
#         noiseVar   = 0,
#         maxWait    = 50
#     )
#     ac = ActorCritic(state_dim=1).to(device)
#     actor_opt  = optim.Adam(ac.actor.parameters(),  lr=ACTOR_LR)
#     critic_opt = optim.Adam(ac.critic.parameters(), lr=CRITIC_LR)
#     lambda_bins = np.linspace(LAMBDA_MIN, LAMBDA_MAX, NUM_LAMBDAS + 1)

#     tqdm_bar = tqdm(range(TOTAL_EPISODES), desc="training")
#     for ep in tqdm_bar:
#         all_s, all_a, all_r, all_v = [], [], [], []
#         all_logp, all_ent, all_lam, all_adv = [], [], [], []

#         for lam_idx in range(NUM_LAMBDAS):
#             lam_val    = np.random.uniform(lambda_bins[lam_idx], lambda_bins[lam_idx+1])
#             lam_tensor = torch.tensor([[lam_val]], dtype=torch.float32, device=device)
#             state = env.reset()
#             # env.arm[0]=np.random.randint(1, 51)
#             # state = np.array([env.arm[0]], dtype=np.float32)
#             done, step = False, 0
#             states, actions, rewards, values, dones, logps, ents = [], [], [], [], [], [], []
#             while (not done) and step < MAX_STEPS_PER_EPISODE and state[0] < env.maxWait:
#                 step += 1
#                 s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

#                 with torch.no_grad():
#                     epsilon = 0.1
#                     idx     = ac.act(s_t)
#                     logits  = idx - lam_tensor
#                     dist    = torch.distributions.Bernoulli(logits=logits)
#                     if torch.rand(1).item() < epsilon:
#                         a_t = torch.bernoulli(torch.full_like(logits, 0.5))
#                     else:
#                         a_t = dist.sample()
#                     a_t = a_t.view(1, 1)
#                     logp = dist.log_prob(a_t)
#                     ent  = dist.entropy()
#                     v_t  = ac.value(s_t, lam_tensor)

#                 reward = env._calReward(int(a_t.item()), state[0])
#                 if a_t.item() == 1:
#                     reward -= lam_val
#                 next_state, _, _, _ = env.step(int(a_t.item()))
#                 states.append(s_t)
#                 actions.append(a_t)
#                 rewards.append(reward)
#                 values.append(v_t.item())
#                 logps.append(logp)
#                 ents.append(ent)
#                 dones.append(done)
#                 state = next_state
#                 if state[0] == 1:
#                     break

#             next_v   = ac.value(
#                 torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
#                 lam_tensor
#             ).item()
#             returns  = compute_gae(rewards, values, next_v, dones)
#             adv      = torch.tensor(returns, device=device) - torch.tensor(values, device=device)
#             all_s   += states
#             all_a   += actions
#             all_r   += returns
#             all_v   += values
#             all_logp+= logps
#             all_ent += ents
#             all_lam += [lam_val] * len(states)
#             all_adv += adv.tolist()

#         # ======== 转成 Tensor ========
#         S   = torch.cat(all_s)                                # (N,1)
#         A   = torch.stack(all_a)                              # (N,1)
#         RET = torch.tensor(all_r, dtype=torch.float32, device=device).unsqueeze(1)
#         V   = torch.tensor(all_v, dtype=torch.float32, device=device).unsqueeze(1)
#         ADV = torch.tensor(all_adv, dtype=torch.float32, device=device).unsqueeze(1)
#         ADV = (ADV - ADV.mean()) / (ADV.std(unbiased=False) + 1e-8)
#         LAM = torch.tensor(all_lam, dtype=torch.float32, device=device).unsqueeze(1)

#         # ======== Critic 更新 ========
#         for _ in range(N_CRITIC_UPDATES):
#             critic_opt.zero_grad()
#             v_pred = ac.value(S, LAM)
#             v_loss = ((v_pred - RET) ** 2).mean()
#             v_loss.backward()
#             nn.utils.clip_grad_norm_(ac.critic.parameters(), GRAD_CLIP_NORM)
#             critic_opt.step()

#         # ======== 构造 Actor BCE target（由Critic判断） ========
#         with torch.no_grad():
#             S_np = S.cpu().numpy()
#             LAM_np = LAM.cpu().numpy()
#             reward1 = []
#             reward0 = []
#             for i in range(len(S_np)):
#                 state_i = S_np[i][0]
#                 lam_i = LAM_np[i][0]
#                 # reward + 下一状态的 value
#                 r1 = env._calReward(1, state_i) - lam_i
#                 r0 = env._calReward(0, state_i)
#                 # 下一个状态已知规则（action=1回1，action=0加1且不超过maxWait）
#                 ns1 = np.array([1.0], dtype=np.float32)
#                 ns0 = np.array([min(state_i+1, env.maxWait)], dtype=np.float32)
#                 s1 = torch.tensor(ns1, dtype=torch.float32, device=device).unsqueeze(0)
#                 s0 = torch.tensor(ns0, dtype=torch.float32, device=device).unsqueeze(0)
#                 v1 = r1 + GAMMA * ac.value(s1, torch.tensor([[lam_i]], dtype=torch.float32, device=device)).item()
#                 v0 = r0 + GAMMA * ac.value(s0, torch.tensor([[lam_i]], dtype=torch.float32, device=device)).item()
#                 reward1.append(v1)
#                 reward0.append(v0)
#             target = (np.array(reward1) > np.array(reward0)).astype(np.float32)
#             target = torch.tensor(target, device=device).unsqueeze(1)

#         # ======== Actor 用 Critic 指导的 BCE Loss 更新 ========
#         actor_opt.zero_grad()
#         idx_pred = ac.act(S)
#         logits   = idx_pred - LAM
#         bce_loss = nn.BCEWithLogitsLoss()(logits, target)
#         loss = bce_loss 
#         loss.backward()
#         nn.utils.clip_grad_norm_(ac.actor.parameters(), GRAD_CLIP_NORM)
#         actor_opt.step()

#         tqdm_bar.set_description(
#             f"Ep {ep+1}/{TOTAL_EPISODES} | Aloss {loss.item():.4f} | Vloss {v_loss.item():.4f}"
#         )

#     return ac

# if __name__ == "__main__":
#     ac_model = train()
#     visualize_index(ac_model)
#endregion




# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# from rc_env import recoveringBanditsEnv   # 你的环境
# from model import ActorCritic            # 你的网络 (需定义 act(), value())

# # ========= 设备选择 =========
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========= 超参数 =========
# GAMMA = 0.99
# GAE_LAMBDA = 0.95
# TOTAL_EPISODES = 10000
# MAX_STEPS_PER_EPISODE = 50

# LAMBDA_MIN, LAMBDA_MAX = 1, 11
# NUM_LAMBDAS = 10

# ACTOR_LR  = 1e-4
# CRITIC_LR = 1e-5
# N_CRITIC_UPDATES = 10
# ENTROPY_COEF = 0.01
# GRAD_CLIP_NORM = 5.0

# def visualize_index(ac, max_state=40):
#     ac.actor.eval()
#     states = torch.arange(1, max_state + 1, dtype=torch.float32).unsqueeze(1).to(device)
#     with torch.no_grad():
#         index_preds = ac.actor(states)
#     plt.figure(figsize=(8, 5))
#     plt.plot(states.cpu().numpy().squeeze(),
#              index_preds.cpu().numpy().squeeze(),
#              label="Whittle Index Estimate", marker='o')
#     plt.xlabel("State")
#     plt.ylabel("Predicted Index")
#     plt.title("Actor Network: Whittle Index vs. State")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def train():
#     env = recoveringBanditsEnv(seed=42, thetaVals=[10, 0.5, 0.0], noiseVar=0, maxWait=50)
#     ac = ActorCritic(state_dim=1).to(device)
#     actor_opt = optim.Adam(ac.actor.parameters(), lr=ACTOR_LR)
#     critic_opt = optim.Adam(ac.critic.parameters(), lr=CRITIC_LR)

#     tqdm_bar = tqdm(range(TOTAL_EPISODES), desc="training")
#     for ep in tqdm_bar:
#         all_s, all_r, all_v = [], [], []
#         all_lam, all_a1_v, all_a0_v = [], [], []

#         lambda_vals = np.random.uniform(LAMBDA_MIN, LAMBDA_MAX, size=NUM_LAMBDAS)
#         for lam_val in lambda_vals:
#             lam_tensor = torch.tensor([[lam_val]], dtype=torch.float32, device=device)

#             state = env.reset()
#             # env.arm[0] = np.random.randint(1, env.maxWait + 1)
#             # state = np.array([env.arm[0]], dtype=np.float32)

#             for step in range(MAX_STEPS_PER_EPISODE):
#                 s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

#                 # ==== 计算 Q(s,1) ====
#                 reward_1 = env._calReward(1, state[0]) - lam_val
#                 q1 = reward_1
#                 env.step(1)
#                 if env.arm[0] != 1:
#                     next_v = ac.value(torch.tensor(env.arm[0], dtype=torch.float32, device=device).view(1, 1), lam_tensor).item()
#                     q1 += GAMMA * next_v
#                 env.arm[0] = state[0]

#                 # ==== 计算 Q(s,0) ====
#                 reward_0 = env._calReward(0, state[0])
#                 q0 = reward_0
#                 env.step(0)
#                 if env.arm[0] != 1:
#                     next_v = ac.value(torch.tensor(env.arm[0], dtype=torch.float32, device=device).view(1, 1), lam_tensor).item()
#                     q0 += GAMMA * next_v
#                 env.arm[0] = state[0]

#                 with torch.no_grad():
#                     idx_pred = ac.act(s_t).item()
#                     pi_1 = torch.sigmoid(torch.tensor([[idx_pred - lam_val]])).item()
#                     pi_0 = 1 - pi_1
#                     v_s = pi_1 * q1 + pi_0 * q0

#                 all_s.append(s_t)
#                 all_v.append(v_s)
#                 all_r.append(reward_1)
#                 all_lam.append(lam_val)
#                 all_a1_v.append(q1 - v_s)
#                 all_a0_v.append(q0 - v_s)

#                 EPSILON = 0.1  # 可以放超参数区，或者逐步衰减
#                 if np.random.rand() < EPSILON:
#                     action = np.random.choice([0, 1])  # 随机探索
#                 else:
#                     action = 1 if pi_1 > pi_0 else 0   # 按照策略选择
#                 _, _, done, _ = env.step(action)
#                 #print("state:", state, "action:", action, "next state:", env.arm[0])
#                 state = np.array([env.arm[0]], dtype=np.float32)
#                 if done or state[0] == 1:
#                     break

#         S = torch.cat(all_s)
#         V = torch.tensor(all_v, dtype=torch.float32, device=device).unsqueeze(1)
#         A1 = torch.tensor(all_a1_v, dtype=torch.float32, device=device).unsqueeze(1)
#         A0 = torch.tensor(all_a0_v, dtype=torch.float32, device=device).unsqueeze(1)
#         LAM = torch.tensor(all_lam, dtype=torch.float32, device=device).unsqueeze(1)

#         ADV = torch.cat([A1, A0], dim=1)
#         ADV_mean, ADV_std = ADV.mean(), ADV.std(unbiased=False)
#         A1 = (A1 - ADV_mean) / (ADV_std + 1e-8)
#         A0 = (A0 - ADV_mean) / (ADV_std + 1e-8)

#         for _ in range(N_CRITIC_UPDATES):
#             critic_opt.zero_grad()
#             v_pred = ac.value(S, LAM)
#             v_loss = ((v_pred - V) ** 2).mean()
#             v_loss.backward()
#             nn.utils.clip_grad_norm_(ac.critic.parameters(), GRAD_CLIP_NORM)
#             critic_opt.step()

#         actor_opt.zero_grad()
#         idx_pred = ac.act(S)
#         logits = idx_pred - LAM
#         prob_1 = torch.sigmoid(logits)
#         prob_0 = 1 - prob_1
#         log_prob_1 = torch.log(prob_1 + 1e-8)
#         log_prob_0 = torch.log(prob_0 + 1e-8)
#         actor_loss = - (prob_1 * log_prob_1 * A1 + prob_0 * log_prob_0 * A0).mean()
#         actor_loss.backward()
#         nn.utils.clip_grad_norm_(ac.actor.parameters(), GRAD_CLIP_NORM)
#         actor_opt.step()

#         tqdm_bar.set_description(
#             f"Ep {ep+1}/{TOTAL_EPISODES} | Aloss {actor_loss.item():.4f} | Vloss {v_loss.item():.4f}"
#         )

#     return ac

# if __name__ == "__main__":
#     ac_model = train()
#     visualize_index(ac_model)

