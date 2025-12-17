import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rc_env import recoveringBanditsEnv   # 你的环境
from model  import ActorCritic            # 你的网络 (需定义 act(), value())

# ========= 设备选择 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= 超参数 =========
GAMMA                = 0.99
GAE_LAMBDA           = 0.95
TOTAL_EPISODES       = 100000
MAX_STEPS_PER_EPISODE= 400

LAMBDA_MIN, LAMBDA_MAX = 1, 11
NUM_LAMBDAS            = 10

ACTOR_LR  = 1e-3
CRITIC_LR = 1e-4
N_CRITIC_UPDATES = 10
ENTROPY_COEF    = 0.0
GRAD_CLIP_NORM  = 5.0

def visualize_index(ac, max_state=40):
    ac.actor.eval()
    states = torch.arange(1, max_state + 1, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        index_preds = ac.actor(states)
    plt.figure(figsize=(8, 5))
    plt.plot(states.cpu().numpy().squeeze(),
             index_preds.cpu().numpy().squeeze(),
             label="Whittle Index Estimate", marker='o')
    plt.xlabel("State")
    plt.ylabel("Predicted Index")
    plt.title("Actor Network: Whittle Index vs. State")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_gae(rewards, values, next_value, dones):
    values = values + [next_value]
    gae, returns = 0.0, []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t+1] * (1 - dones[t]) - values[t]
        gae   = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
        returns.insert(0, gae + values[t])
    return returns

def set_seed(seed=114514):
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def train(seed=114514):
    set_seed(seed)
    env = recoveringBanditsEnv(
        seed       = seed,
        thetaVals  = [10, 0.5, 0.0],
        noiseVar   = 0,
        maxWait    = 50
    )
    ac = ActorCritic(state_dim=1).to(device)
    actor_opt  = optim.Adam(ac.actor.parameters(),  lr=ACTOR_LR)
    critic_opt = optim.Adam(ac.critic.parameters(), lr=CRITIC_LR)
    lambda_bins = np.linspace(LAMBDA_MIN, LAMBDA_MAX, NUM_LAMBDAS + 1)

    tqdm_bar = tqdm(range(TOTAL_EPISODES), desc="training")
    for ep in tqdm_bar:
        all_s, all_a, all_r, all_v = [], [], [], []
        all_logp, all_ent, all_lam, all_adv = [], [], [], []

        for lam_idx in range(NUM_LAMBDAS):
            lam_val    = np.random.uniform(lambda_bins[lam_idx], lambda_bins[lam_idx+1])
            lam_tensor = torch.tensor([[lam_val]], dtype=torch.float32, device=device)
            state = env.reset()
            # env.arm[0]=np.random.randint(1, 51)
            # state = np.array([env.arm[0]], dtype=np.float32)
            done, step = False, 0
            states, actions, rewards, values, dones, logps, ents = [], [], [], [], [], [], []
            while (not done) and step < MAX_STEPS_PER_EPISODE and state[0] < env.maxWait:
                step += 1
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    epsilon = 0.1
                    idx     = ac.act(s_t)
                    logits  = idx - lam_tensor
                    dist    = torch.distributions.Bernoulli(logits=logits)
                    if torch.rand(1).item() < epsilon:
                        a_t = torch.bernoulli(torch.full_like(logits, 0.5))
                    else:
                        a_t = dist.sample()
                    a_t = a_t.view(1, 1)
                    logp = dist.log_prob(a_t)
                    ent  = dist.entropy()
                    v_t  = ac.value(s_t, lam_tensor)

                reward = env._calReward(int(a_t.item()), state[0])
                if a_t.item() == 1:
                    reward -= lam_val
                next_state, _, _, _ = env.step(int(a_t.item()))
                states.append(s_t)
                actions.append(a_t)
                rewards.append(reward)
                values.append(v_t.item())
                logps.append(logp)
                ents.append(ent)
                dones.append(done)
                state = next_state
                if state[0] == 1:
                    break

            next_v   = ac.value(
                torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                lam_tensor
            ).item()
            returns  = compute_gae(rewards, values, next_v, dones)
            adv      = torch.tensor(returns, device=device) - torch.tensor(values, device=device)
            all_s   += states
            all_a   += actions
            all_r   += returns
            all_v   += values
            all_logp+= logps
            all_ent += ents
            all_lam += [lam_val] * len(states)
            all_adv += adv.tolist()

        # ======== 转成 Tensor ========
        S   = torch.cat(all_s)                                # (N,1)
        A   = torch.stack(all_a)                              # (N,1)
        RET = torch.tensor(all_r, dtype=torch.float32, device=device).unsqueeze(1)
        V   = torch.tensor(all_v, dtype=torch.float32, device=device).unsqueeze(1)
        ADV = torch.tensor(all_adv, dtype=torch.float32, device=device).unsqueeze(1)
        ADV = (ADV - ADV.mean()) / (ADV.std(unbiased=False) + 1e-8)
        LAM = torch.tensor(all_lam, dtype=torch.float32, device=device).unsqueeze(1)

        # ======== Critic 更新 ========
        for _ in range(N_CRITIC_UPDATES):
            critic_opt.zero_grad()
            v_pred = ac.value(S, LAM)
            v_loss = ((v_pred - RET) ** 2).mean()
            v_loss.backward()
            nn.utils.clip_grad_norm_(ac.critic.parameters(), GRAD_CLIP_NORM)
            critic_opt.step()

        # ======== 构造 Actor BCE target（由Critic判断） ========
        with torch.no_grad():
            S_np = S.cpu().numpy()
            LAM_np = LAM.cpu().numpy()
            reward1 = []
            reward0 = []
            for i in range(len(S_np)):
                state_i = S_np[i][0]
                lam_i = LAM_np[i][0]
                # reward + 下一状态的 value
                r1 = env._calReward(1, state_i) - lam_i
                r0 = env._calReward(0, state_i)
                # 下一个状态已知规则（action=1回1，action=0加1且不超过maxWait）
                ns1 = np.array([1.0], dtype=np.float32)
                ns0 = np.array([min(state_i+1, env.maxWait)], dtype=np.float32)
                s1 = torch.tensor(ns1, dtype=torch.float32, device=device).unsqueeze(0)
                s0 = torch.tensor(ns0, dtype=torch.float32, device=device).unsqueeze(0)
                v1 = r1 + GAMMA * ac.value(s1, torch.tensor([[lam_i]], dtype=torch.float32, device=device)).item()
                v0 = r0 + GAMMA * ac.value(s0, torch.tensor([[lam_i]], dtype=torch.float32, device=device)).item()
                reward1.append(v1)
                reward0.append(v0)
            target = (np.array(reward1) > np.array(reward0)).astype(np.float32)
            target = torch.tensor(target, device=device).unsqueeze(1)

        # ======== Actor 用 Critic 指导的 BCE Loss 更新 ========
        actor_opt.zero_grad()
        idx_pred = ac.act(S)
        logits   = idx_pred - LAM
        bce_loss = nn.BCEWithLogitsLoss()(logits, target)
        prob = torch.sigmoid(logits)
        entropy = - (prob * torch.log(prob + 1e-8) + (1 - prob) * torch.log(1 - prob + 1e-8))
        loss = bce_loss - ENTROPY_COEF * entropy.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(ac.actor.parameters(), GRAD_CLIP_NORM)
        actor_opt.step()

        tqdm_bar.set_description(
            f"Ep {ep+1}/{TOTAL_EPISODES} | Aloss {loss.item():.4f} | Vloss {v_loss.item():.4f}"
        )

    return ac

if __name__ == "__main__":
    ac_model = train()
    visualize_index(ac_model)





#region
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# from rc_env import recoveringBanditsEnv   # 环境类
# from model  import ActorCritic            # 网络结构

# # ========= 设备选择 =========
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========= 超参数 =========
# GAMMA                = 0.99
# GAE_LAMBDA           = 0.95
# TOTAL_EPISODES       = 10000
# MAX_STEPS_PER_EPISODE= 400

# LAMBDA_MIN, LAMBDA_MAX = 1, 11
# NUM_LAMBDAS            = 10
# WARMUP_RATIO           = 0.3

# ACTOR_LR  = 1e-3
# CRITIC_LR = 1e-4
# N_CRITIC_UPDATES = 10
# ENTROPY_COEF    = 0.01
# GRAD_CLIP_NORM  = 5.0

# # ========= 可视化 =========
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

# # ========= GAE =========
# def compute_gae(rewards, values, next_value, dones):
#     values = values + [next_value]
#     gae, returns = 0.0, []
#     for t in reversed(range(len(rewards))):
#         delta = rewards[t] + GAMMA * values[t+1] * (1 - dones[t]) - values[t]
#         gae   = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
#         returns.insert(0, gae + values[t])
#     return returns

# # ========= λ采样函数 =========
# def sample_lambdas(ac, warmup_done, num_lambdas, lambda_bins, device):
#     if warmup_done:
#         states = torch.randint(1, 51, (num_lambdas, 1), dtype=torch.float32, device=device)
#         with torch.no_grad():
#             lambdas = ac.actor(states).squeeze().cpu().numpy().tolist()
#     else:
#         lambdas = [np.random.uniform(lambda_bins[i], lambda_bins[i+1]) for i in range(num_lambdas)]
#     return lambdas

# # ========= 训练主函数 =========
# def train():
#     env = recoveringBanditsEnv(seed=42, thetaVals=[10, 0.5, 0.0], noiseVar=0, maxWait=50)
#     ac = ActorCritic(state_dim=1).to(device)
#     actor_opt  = optim.Adam(ac.actor.parameters(),  lr=ACTOR_LR)
#     critic_opt = optim.Adam(ac.critic.parameters(), lr=CRITIC_LR)
#     lambda_bins = np.linspace(LAMBDA_MIN, LAMBDA_MAX, NUM_LAMBDAS + 1)

#     tqdm_bar = tqdm(range(TOTAL_EPISODES), desc="training")
#     for ep in tqdm_bar:
#         warmup_done = (ep / TOTAL_EPISODES) >= WARMUP_RATIO
#         lambda_vals = sample_lambdas(ac, warmup_done, NUM_LAMBDAS, lambda_bins, device)

#         all_s, all_a, all_r, all_v = [], [], [], []
#         all_logp, all_ent, all_lam, all_adv = [], [], [], []

#         for lam_val in lambda_vals:
#             lam_tensor = torch.tensor([[lam_val]], dtype=torch.float32, device=device)
#             state = env.reset()
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

#             next_v = ac.value(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0), lam_tensor).item()
#             returns = compute_gae(rewards, values, next_v, dones)
#             adv     = torch.tensor(returns, device=device) - torch.tensor(values, device=device)

#             all_s   += states
#             all_a   += actions
#             all_r   += returns
#             all_v   += values
#             all_logp+= logps
#             all_ent += ents
#             all_lam += [lam_val] * len(states)
#             all_adv += adv.tolist()

#         S   = torch.cat(all_s)
#         A   = torch.stack(all_a)
#         RET = torch.tensor(all_r, dtype=torch.float32, device=device).unsqueeze(1)
#         V   = torch.tensor(all_v, dtype=torch.float32, device=device).unsqueeze(1)
#         ADV = torch.tensor(all_adv, dtype=torch.float32, device=device).unsqueeze(1)
#         ADV = (ADV - ADV.mean()) / (ADV.std(unbiased=False) + 1e-8)
#         LAM = torch.tensor(all_lam, dtype=torch.float32, device=device).unsqueeze(1)

#         for _ in range(N_CRITIC_UPDATES):
#             critic_opt.zero_grad()
#             v_pred = ac.value(S, LAM)
#             v_loss = ((v_pred - RET) ** 2).mean()
#             v_loss.backward()
#             nn.utils.clip_grad_norm_(ac.critic.parameters(), GRAD_CLIP_NORM)
#             critic_opt.step()

#         if ep % N_CRITIC_UPDATES == 0:
#             actor_opt.zero_grad()
#             idx_logits = (ac.act(S) - LAM) * 1
#             dist       = torch.distributions.Bernoulli(logits=idx_logits)
#             new_logp   = dist.log_prob(A)
#             pg_loss    = -(new_logp * ADV).mean()
#             actor_loss = pg_loss
#             actor_loss.backward()
#             nn.utils.clip_grad_norm_(ac.actor.parameters(), GRAD_CLIP_NORM)
#             actor_opt.step()

#             tqdm_bar.set_description(
#                 f"Ep {ep+1}/{TOTAL_EPISODES} | Aloss {actor_loss.item():.4f} | Vloss {v_loss.item():.4f}"
#             )

#     return ac

# if __name__ == "__main__":
#     ac_model = train()
#     visualize_index(ac_model)








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
# TOTAL_EPISODES       = 10000     # 设成 100_000 可以得到更好效果
# MAX_STEPS_PER_EPISODE= 400

# # Whittle λ 采样区间
# LAMBDA_MIN, LAMBDA_MAX = 1, 11
# NUM_LAMBDAS            = 10         # 每轮采 10 个 λ

# # 学习率与更新频率
# ACTOR_LR  = 1e-3
# CRITIC_LR = 1e-4
# N_CRITIC_UPDATES = 10  # 每次 Actor 更新前 Critic 更新次数
# ENTROPY_COEF    = 0.01
# GRAD_CLIP_NORM  = 5.0               # 可选：梯度裁剪

# # ================= 可视化 =================
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

# # ================= GAE =================
# def compute_gae(rewards, values, next_value, dones):
#     values = values + [next_value]      # 长度 T+1
#     gae, returns = 0.0, []
#     for t in reversed(range(len(rewards))):
#         delta = rewards[t] + GAMMA * values[t+1] * (1 - dones[t]) - values[t]
#         gae   = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
#         returns.insert(0, gae + values[t])
#     return returns

# # ================= 训练 =================
# def train():
#     # ---------- 创建环境 ----------
#     env = recoveringBanditsEnv(
#         seed       = 42,
#         thetaVals  = [10, 0.5, 0.0],
#         noiseVar   = 0,
#         maxWait    = 50
#     )

#     # ---------- 创建网络 ----------
#     ac = ActorCritic(state_dim=1).to(device)
#     actor_opt  = optim.Adam(ac.actor.parameters(),  lr=ACTOR_LR)
#     critic_opt = optim.Adam(ac.critic.parameters(), lr=CRITIC_LR)

#     # ---------- λ 采样区间 ----------
#     lambda_bins = np.linspace(LAMBDA_MIN, LAMBDA_MAX, NUM_LAMBDAS + 1)

#     tqdm_bar = tqdm(range(TOTAL_EPISODES), desc="training")
#     for ep in tqdm_bar:
#         # ======== 批次缓存 ========
#         all_s, all_a, all_r, all_v = [], [], [], []
#         all_logp, all_ent, all_lam, all_adv = [], [], [], []

#         # ======== 采集 NUM_LAMBDAS 条轨迹 ========
#         for lam_idx in range(NUM_LAMBDAS):
#             lam_val    = np.random.uniform(lambda_bins[lam_idx], lambda_bins[lam_idx+1])
#             lam_tensor = torch.tensor([[lam_val]], dtype=torch.float32, device=device)

#             # 随机起点（等待时间）
#             state = env.reset()
#             done, step = False, 0

#             # --- 单条 trajectory 缓存 ---
#             states, actions, rewards, values, dones, logps, ents = [], [], [], [], [], [], []
#             while (not done) and step < MAX_STEPS_PER_EPISODE and state[0] < env.maxWait:
#                 step += 1
#                 s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

#                 # ---------- Actor 采样动作 (logits 版本) ----------
#                 with torch.no_grad():
#                     epsilon = 0.1  # ε的值，可设为 config 或逐渐递减

#                     idx     = ac.act(s_t)                 # Whittle index
#                     logits  = idx - lam_tensor            # π(a=1|s)=σ(logits)
#                     dist    = torch.distributions.Bernoulli(logits=logits)
#                     if torch.rand(1).item() < epsilon:
#                         # 用均匀分布产生随机动作，保持形状一致 [1,1]
#                         a_t = torch.bernoulli(torch.full_like(logits, 0.5))
#                     else:
#                         a_t = dist.sample()
#                     # 统一 shape 成 [1,1]
#                     a_t = a_t.view(1, 1)
#                     logp = dist.log_prob(a_t)
#                     ent  = dist.entropy()
#                     v_t  = ac.value(s_t, lam_tensor)      # Critic

#                 # ---------- 环境一步 ----------
#                 reward = env._calReward(int(a_t.item()), state[0])
#                 if a_t.item() == 1:
#                     reward -= lam_val                     # sub λ

#                 next_state, _, _, _ = env.step(int(a_t.item()))
#                 #print(f"Step {step}: State {state[0]} -> Action {int(a_t.item())} | Reward {reward:.2f} | Next State {next_state[0]} | λ {lam_val:.2f}")  
#                 # ---------- 缓存 ----------
#                 states.append(s_t)
#                 actions.append(a_t)
#                 rewards.append(reward)
#                 values.append(v_t.item())
#                 logps.append(logp)
#                 ents.append(ent)
#                 dones.append(done)

#                 state = next_state
#                 if state[0] == 1:                         # 重置即结束
#                     break

#             # --- 计算 GAE / returns & advantage ---
#             next_v   = ac.value(
#                 torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
#                 lam_tensor
#             ).item()
#             returns  = compute_gae(rewards, values, next_v, dones)
#             adv      = torch.tensor(returns, device=device) - torch.tensor(values, device=device)

#             # --- 批量合并 ---
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

#         # ======== 更新 Critic ========
#         for _ in range(N_CRITIC_UPDATES):
#             critic_opt.zero_grad()
#             v_pred = ac.value(S, LAM)
#             v_loss = ((v_pred - RET) ** 2).mean()
#             v_loss.backward()
#             nn.utils.clip_grad_norm_(ac.critic.parameters(), GRAD_CLIP_NORM)
#             critic_opt.step()

#         # ======== 更新 Actor ========
#         if ep % N_CRITIC_UPDATES == 0:
#             actor_opt.zero_grad()
#             idx_logits = (ac.act(S) - LAM) * 100       # logits 批
#             dist       = torch.distributions.Bernoulli(logits=idx_logits)
#             new_logp   = dist.log_prob(A)              # shape: (N,1)

#             # ===== 策略梯度项 =====
#             raw_loss = -new_logp * ADV                 # shape: (N,1)

#             # ===== 筛除部分负 loss（仅在 loss < 0 且满足概率门槛时置为 0）=====
#             mask = torch.ones_like(raw_loss)
#             negative_mask = raw_loss > 0
#             skip_mask = torch.rand_like(raw_loss) < (ep*0.9/TOTAL_EPISODES)   # 30% 概率跳过负 loss
#             final_mask = ~(negative_mask & skip_mask)     # False 的位置将会被置 0

#             # 应用 mask
#             masked_loss = raw_loss * final_mask.float()

#             pg_loss = masked_loss.mean()

#             # ===== 可选：熵正则 =====
#             ent_reg    = -ENTROPY_COEF * dist.entropy().mean()
#             actor_loss = pg_loss  # + ent_reg（如需可加回）

#             actor_loss.backward()
#             nn.utils.clip_grad_norm_(ac.actor.parameters(), GRAD_CLIP_NORM)
#             actor_opt.step()

# #         if(ep% N_CRITIC_UPDATES == 0):
# #             actor_opt.zero_grad()
# #             idx_logits = (ac.act(S) - LAM)*100       # logits 批
# #             dist       = torch.distributions.Bernoulli(logits=idx_logits)
# #             new_logp   = dist.log_prob(A)

# #             pg_loss    = -(new_logp * ADV).mean()
# #             #pg_loss    = (new_logp * ADV).mean()
# #             ent_reg    = -ENTROPY_COEF * dist.entropy().mean()
# # #            actor_loss = pg_loss + ent_reg
# #             actor_loss = pg_loss
# #             actor_loss.backward()
# #             nn.utils.clip_grad_norm_(ac.actor.parameters(), GRAD_CLIP_NORM)
# #             actor_opt.step()

#             tqdm_bar.set_description(
#                 f"Ep {ep+1}/{TOTAL_EPISODES} | Aloss {actor_loss.item():.4f} | Vloss {v_loss.item():.4f}"
#             )

#     return ac

# # ================= 运行 =================
# if __name__ == "__main__":
#     ac_model = train()
#     visualize_index(ac_model)
#endregion
