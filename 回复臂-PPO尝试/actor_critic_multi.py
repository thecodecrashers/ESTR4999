import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========= 引入环境和模型 =========
from multi_env import get_10_arm_env, get_20_arm_env
from model import ActorCritic

# ========= 设备选择 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= 超参数 =========
GAMMA = 0.99
GAE_LAMBDA = 0.95
TOTAL_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 100
WQRMUP = 0.3

ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
N_CRITIC_UPDATES = 10
ENTROPY_COEF = 0.0
GRAD_CLIP_NORM = 5.0

LAMBDA_MIN, LAMBDA_MAX = 1, 11
NUM_LAMBDAS = 80

# ========= 工具函数 =========
def compute_gae(rewards, values, next_value, dones):
    values = values + [next_value]
    gae, returns = 0.0, []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
        returns.insert(0, gae + values[t])
    return returns

def set_seed(seed=114514):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# ========= 主训练函数（单个 arm） =========
def train_single_arm(arm_name, env, env_name, save_dir):
    subdir = os.path.join(save_dir, f"{env_name}_{arm_name}")
    os.makedirs(subdir, exist_ok=True)

    ac = ActorCritic(state_dim=1).to(device)
    actor_opt = optim.Adam(ac.actor.parameters(), lr=ACTOR_LR)
    critic_opt = optim.Adam(ac.critic.parameters(), lr=CRITIC_LR)
    lambda_bins = np.linspace(LAMBDA_MIN, LAMBDA_MAX, NUM_LAMBDAS + 1)

    loss_log = []
    vloss_log = []
    tqdm_bar = tqdm(range(TOTAL_EPISODES), desc="training")
    for ep in tqdm_bar:
        all_s, all_a, all_r, all_v = [], [], [], []
        all_logp, all_ent, all_lam, all_adv = [], [], [], []

        for lam_idx in range(NUM_LAMBDAS):
            if ep < TOTAL_EPISODES * WQRMUP:
                lam_val = np.random.uniform(lambda_bins[lam_idx], lambda_bins[lam_idx + 1])
            else:
                hh = np.random.randint(0, 51)
                with torch.no_grad():
                    lam_val = ac.actor(torch.tensor([[hh]], dtype=torch.float32, device=device)).item()
                    if lam_val<-0.01:
                        lam_val=0
            lam_tensor = torch.tensor([[lam_val]], dtype=torch.float32, device=device)

            state = env[arm_name].reset()
            done, step = False, 0
            states, actions, rewards, values, dones, logps, ents = [], [], [], [], [], [], []

            while (not done) and step < MAX_STEPS_PER_EPISODE and state[0] < env[arm_name].maxWait:
                step += 1
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    epsilon = 0.2
                    idx = ac.act(s_t)
                    logits = idx - lam_tensor
                    dist = torch.distributions.Bernoulli(logits=logits)
                    if torch.rand(1).item() < epsilon:
                        a_t = torch.bernoulli(torch.full_like(logits, 0.5))
                        #a_t=torch.bernoulli(torch.full_like(logits, 0))
                    else:
                        a_t = dist.sample()
                    a_t = a_t.view(1, 1)
                    logp = dist.log_prob(a_t)
                    ent = dist.entropy()
                    v_t = ac.value(s_t, lam_tensor)

                reward = env[arm_name]._calReward(int(a_t.item()), state[0])
                if a_t.item() == 1:
                    reward -= lam_val
                next_state, _, _, _ = env[arm_name].step(int(a_t.item()))
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

            next_v = ac.value(
                torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
                lam_tensor
            ).item()
            returns = compute_gae(rewards, values, next_v, dones)
            adv = torch.tensor(returns, device=device) - torch.tensor(values, device=device)
            all_s += states
            all_a += actions
            all_r += returns
            all_v += values
            all_logp += logps
            all_ent += ents
            all_lam += [lam_val] * len(states)
            all_adv += adv.tolist()

        S = torch.cat(all_s)
        A = torch.stack(all_a)
        RET = torch.tensor(all_r, dtype=torch.float32, device=device).unsqueeze(1)
        V = torch.tensor(all_v, dtype=torch.float32, device=device).unsqueeze(1)
        ADV = torch.tensor(all_adv, dtype=torch.float32, device=device).unsqueeze(1)
        ADV = (ADV - ADV.mean()) / (ADV.std(unbiased=False) + 1e-8)
        LAM = torch.tensor(all_lam, dtype=torch.float32, device=device).unsqueeze(1)

        for _ in range(N_CRITIC_UPDATES):
            critic_opt.zero_grad()
            v_pred = ac.value(S, LAM)
            v_loss = ((v_pred - RET) ** 2).mean()
            v_loss.backward()
            nn.utils.clip_grad_norm_(ac.critic.parameters(), GRAD_CLIP_NORM)
            critic_opt.step()

        with torch.no_grad():
            S_np = S.cpu().numpy()
            LAM_np = LAM.cpu().numpy()
            reward1, reward0 = [], []
            for i in range(len(S_np)):
                state_i = S_np[i][0]
                lam_i = LAM_np[i][0]
                r1 = env[arm_name]._calReward(1, state_i) - lam_i
                r0 = env[arm_name]._calReward(0, state_i)
                ns1 = np.array([1.0], dtype=np.float32)
                ns0 = np.array([min(state_i + 1, env[arm_name].maxWait)], dtype=np.float32)
                s1 = torch.tensor(ns1, dtype=torch.float32, device=device).unsqueeze(0)
                s0 = torch.tensor(ns0, dtype=torch.float32, device=device).unsqueeze(0)
                v1 = r1 + GAMMA * ac.value(s1, torch.tensor([[lam_i]], dtype=torch.float32, device=device)).item()
                v0 = r0 + GAMMA * ac.value(s0, torch.tensor([[lam_i]], dtype=torch.float32, device=device)).item()
                reward1.append(v1)
                reward0.append(v0)
            target = (np.array(reward1) > np.array(reward0)).astype(np.float32)
            target = torch.tensor(target, device=device).unsqueeze(1)

        actor_opt.zero_grad()
        idx_pred = ac.act(S)
        logits = idx_pred - LAM
        bce_loss = nn.BCEWithLogitsLoss()(logits, target)
        prob = torch.sigmoid(logits)
        entropy = - (prob * torch.log(prob + 1e-8) + (1 - prob) * torch.log(1 - prob + 1e-8))
        loss = bce_loss - ENTROPY_COEF * entropy.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(ac.actor.parameters(), GRAD_CLIP_NORM)
        actor_opt.step()

        loss_log.append(loss.item())
        vloss_log.append(v_loss.item())
        tqdm_bar.set_description(
            f"Ep {ep+1}/{TOTAL_EPISODES} | Aloss {loss.item():.4f} | Vloss {v_loss.item():.4f}"
        )

    # ===== 保存模型与日志 =====
    torch.save(ac.state_dict(), os.path.join(subdir, "actor_critic.pth"))
    np.save(os.path.join(subdir, "actor_loss.npy"), np.array(loss_log))
    np.save(os.path.join(subdir, "critic_loss.npy"), np.array(vloss_log))

    with torch.no_grad():
        s_states = torch.arange(1, env[arm_name].maxWait + 1, dtype=torch.float32).unsqueeze(1).to(device)
        whittle_index = ac.actor(s_states).cpu().numpy()
        np.save(os.path.join(subdir, "whittle_index.npy"), whittle_index)

    # ===== 绘图保存 =====
    # Whittle index 曲线
    plt.figure()
    plt.plot(range(1, env[arm_name].maxWait + 1), whittle_index, label="Whittle Index")
    plt.xlabel("State")
    plt.ylabel("Whittle Index")
    plt.title(f"{env_name}_{arm_name} - Whittle Index")
    plt.grid(True)
    plt.savefig(os.path.join(subdir, "whittle_index_plot.png"))
    plt.close()

    # Loss 曲线
    plt.figure()
    plt.plot(loss_log, label="Actor Loss")
    plt.plot(vloss_log, label="Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{env_name}_{arm_name} - Loss Curve")
    plt.grid(True)
    plt.savefig(os.path.join(subdir, "loss_plot.png"))
    plt.close()

    print(f"✅ Finished training {env_name}_{arm_name}")

# ========= 多 arm 多线程训练 =========
def train_multi_arm(env_name, env_getter, save_dir="actor_critic", max_threads=4):
    set_seed(114514)
    os.makedirs(save_dir, exist_ok=True)

    env = env_getter()
    arm_names = list(env.keys())

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(train_single_arm, arm_name, env, env_name, save_dir) for arm_name in arm_names]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error during training: {e}")

# ========= 入口点 =========
if __name__ == "__main__":
    train_multi_arm("10arms", get_10_arm_env, max_threads=1)
    train_multi_arm("20arms", get_20_arm_env, max_threads=1)







# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from multi_env import get_10_arm_env, get_20_arm_env
# from model import ActorCritic

# # ========= 设备选择 =========
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========= 超参数 =========
# GAMMA = 0.99
# GAE_LAMBDA = 0.95
# TOTAL_EPISODES = 1000
# MAX_STEPS_PER_EPISODE = 400
# WQRMUP=0.3

# ACTOR_LR = 1e-3
# CRITIC_LR = 1e-4
# N_CRITIC_UPDATES = 10
# ENTROPY_COEF = 0.0
# GRAD_CLIP_NORM = 5.0

# LAMBDA_MIN, LAMBDA_MAX = 1, 11
# NUM_LAMBDAS = 80


# def compute_gae(rewards, values, next_value, dones):
#     values = values + [next_value]
#     gae, returns = 0.0, []
#     for t in reversed(range(len(rewards))):
#         delta = rewards[t] + GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
#         gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
#         returns.insert(0, gae + values[t])
#     return returns


# def set_seed(seed=114514):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ["PYTHONHASHSEED"] = str(seed)


# def train_single_arm(arm_name, env, env_name, save_dir):
#     subdir = os.path.join(save_dir, f"{env_name}_{arm_name}")
#     os.makedirs(subdir, exist_ok=True)

#     ac = ActorCritic(state_dim=1).to(device)
#     actor_opt = optim.Adam(ac.actor.parameters(), lr=ACTOR_LR)
#     critic_opt = optim.Adam(ac.critic.parameters(), lr=CRITIC_LR)
#     lambda_bins = np.linspace(LAMBDA_MIN, LAMBDA_MAX, NUM_LAMBDAS + 1)

#     loss_log = []
#     vloss_log = []

#     for ep in range(TOTAL_EPISODES):
#         all_s, all_a, all_r, all_v = [], [], [], []
#         all_logp, all_ent, all_lam, all_adv = [], [], [], []

#         for lam_idx in range(NUM_LAMBDAS):
#             if( ep < TOTAL_EPISODES * WQRMUP):
#                 lam_val = np.random.uniform(lambda_bins[lam_idx], lambda_bins[lam_idx + 1])
#             else:
#                 hh=np.randint(0, 51)
#                 with torch.no_grad():
#                     lam_val = ac.actor(torch.tensor([[hh]], dtype=torch.float32, device=device)).item()
#             lam_tensor = torch.tensor([[lam_val]], dtype=torch.float32, device=device)

#             state = env[arm_name].reset()
#             done, step = False, 0
#             states, actions, rewards, values, dones, logps, ents = [], [], [], [], [], [], []

#             while (not done) and step < MAX_STEPS_PER_EPISODE and state[0] < env[arm_name].maxWait:
#                 step += 1
#                 s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

#                 with torch.no_grad():
#                     epsilon = 0.2
#                     idx = ac.act(s_t)
#                     logits = idx - lam_tensor
#                     dist = torch.distributions.Bernoulli(logits=logits)
#                     if torch.rand(1).item() < epsilon:
#                         a_t = torch.bernoulli(torch.full_like(logits, 0.5))
#                     else:
#                         a_t = dist.sample()
#                     a_t = a_t.view(1, 1)
#                     logp = dist.log_prob(a_t)
#                     ent = dist.entropy()
#                     v_t = ac.value(s_t, lam_tensor)

#                 reward = env[arm_name]._calReward(int(a_t.item()), state[0])
#                 if a_t.item() == 1:
#                     reward -= lam_val
#                 next_state, _, _, _ = env[arm_name].step(int(a_t.item()))
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

#             next_v = ac.value(
#                 torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
#                 lam_tensor
#             ).item()
#             returns = compute_gae(rewards, values, next_v, dones)
#             adv = torch.tensor(returns, device=device) - torch.tensor(values, device=device)
#             all_s += states
#             all_a += actions
#             all_r += returns
#             all_v += values
#             all_logp += logps
#             all_ent += ents
#             all_lam += [lam_val] * len(states)
#             all_adv += adv.tolist()

#         S = torch.cat(all_s)
#         A = torch.stack(all_a)
#         RET = torch.tensor(all_r, dtype=torch.float32, device=device).unsqueeze(1)
#         V = torch.tensor(all_v, dtype=torch.float32, device=device).unsqueeze(1)
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

#         with torch.no_grad():
#             S_np = S.cpu().numpy()
#             LAM_np = LAM.cpu().numpy()
#             reward1, reward0 = [], []
#             for i in range(len(S_np)):
#                 state_i = S_np[i][0]
#                 lam_i = LAM_np[i][0]
#                 r1 = env[arm_name]._calReward(1, state_i) - lam_i
#                 r0 = env[arm_name]._calReward(0, state_i)
#                 ns1 = np.array([1.0], dtype=np.float32)
#                 ns0 = np.array([min(state_i + 1, env[arm_name].maxWait)], dtype=np.float32)
#                 s1 = torch.tensor(ns1, dtype=torch.float32, device=device).unsqueeze(0)
#                 s0 = torch.tensor(ns0, dtype=torch.float32, device=device).unsqueeze(0)
#                 v1 = r1 + GAMMA * ac.value(s1, torch.tensor([[lam_i]], dtype=torch.float32, device=device)).item()
#                 v0 = r0 + GAMMA * ac.value(s0, torch.tensor([[lam_i]], dtype=torch.float32, device=device)).item()
#                 reward1.append(v1)
#                 reward0.append(v0)
#             target = (np.array(reward1) > np.array(reward0)).astype(np.float32)
#             target = torch.tensor(target, device=device).unsqueeze(1)

#         actor_opt.zero_grad()
#         idx_pred = ac.act(S)
#         logits = idx_pred - LAM
#         bce_loss = nn.BCEWithLogitsLoss()(logits, target)
#         prob = torch.sigmoid(logits)
#         entropy = - (prob * torch.log(prob + 1e-8) + (1 - prob) * torch.log(1 - prob + 1e-8))
#         loss = bce_loss - ENTROPY_COEF * entropy.mean()
#         loss.backward()
#         nn.utils.clip_grad_norm_(ac.actor.parameters(), GRAD_CLIP_NORM)
#         actor_opt.step()

#         loss_log.append(loss.item())
#         vloss_log.append(v_loss.item())

#     torch.save(ac.state_dict(), os.path.join(subdir, "actor_critic.pth"))
#     np.save(os.path.join(subdir, "actor_loss.npy"), np.array(loss_log))
#     np.save(os.path.join(subdir, "critic_loss.npy"), np.array(vloss_log))

#     with torch.no_grad():
#         s_states = torch.arange(1, env[arm_name].maxWait + 1, dtype=torch.float32).unsqueeze(1).to(device)
#         whittle_index = ac.actor(s_states).cpu().numpy()
#         np.save(os.path.join(subdir, "whittle_index.npy"), whittle_index)

#     print(f"✅ Finished training {env_name}_{arm_name}")


# def train_multi_arm(env_name, env_getter, save_dir="actor_critic", max_threads=4):
#     set_seed(114514)
#     os.makedirs(save_dir, exist_ok=True)

#     env = env_getter()
#     arm_names = list(env.keys())

#     with ThreadPoolExecutor(max_workers=max_threads) as executor:
#         futures = [executor.submit(train_single_arm, arm_name, env, env_name, save_dir) for arm_name in arm_names]
#         for future in as_completed(futures):
#             try:
#                 future.result()
#             except Exception as e:
#                 print(f"❌ Error during training: {e}")


# if __name__ == "__main__":
#     train_multi_arm("10arms", get_10_arm_env, max_threads=4)
#     train_multi_arm("20arms", get_20_arm_env, max_threads=4)






# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from tqdm import tqdm
# from multi_env import get_10_arm_env, get_20_arm_env
# from model import ActorCritic

# # ========= 设备选择 =========
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========= 超参数 =========
# GAMMA = 0.99
# GAE_LAMBDA = 0.95
# TOTAL_EPISODES = 1000
# MAX_STEPS_PER_EPISODE = 400

# ACTOR_LR = 1e-3
# CRITIC_LR = 1e-4
# N_CRITIC_UPDATES = 10
# ENTROPY_COEF = 0.0
# GRAD_CLIP_NORM = 5.0

# LAMBDA_MIN, LAMBDA_MAX = 1, 11
# NUM_LAMBDAS = 80


# def compute_gae(rewards, values, next_value, dones):
#     values = values + [next_value]
#     gae, returns = 0.0, []
#     for t in reversed(range(len(rewards))):
#         delta = rewards[t] + GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
#         gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
#         returns.insert(0, gae + values[t])
#     return returns


# def set_seed(seed=114514):
#     import random
#     import os
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ["PYTHONHASHSEED"] = str(seed)


# def train_multi_arm(env_name, env_getter, save_dir="actor_critic"):
#     set_seed(114514)
#     os.makedirs(save_dir, exist_ok=True)

#     env = env_getter()
#     arm_names = list(env.keys())
#     result = {}

#     for arm_name in arm_names:
#         subdir = os.path.join(save_dir, f"{env_name}_{arm_name}")
#         os.makedirs(subdir, exist_ok=True)

#         ac = ActorCritic(state_dim=1).to(device)
#         actor_opt = optim.Adam(ac.actor.parameters(), lr=ACTOR_LR)
#         critic_opt = optim.Adam(ac.critic.parameters(), lr=CRITIC_LR)
#         lambda_bins = np.linspace(LAMBDA_MIN, LAMBDA_MAX, NUM_LAMBDAS + 1)

#         loss_log = []
#         vloss_log = []

#         tqdm_bar = tqdm(range(TOTAL_EPISODES), desc=f"Training {arm_name}")
#         for ep in tqdm_bar:
#             all_s, all_a, all_r, all_v = [], [], [], []
#             all_logp, all_ent, all_lam, all_adv = [], [], [], []

#             for lam_idx in range(NUM_LAMBDAS):
#                 lam_val = np.random.uniform(lambda_bins[lam_idx], lambda_bins[lam_idx + 1])
#                 lam_tensor = torch.tensor([[lam_val]], dtype=torch.float32, device=device)

#                 state = env[arm_name].reset()
#                 done, step = False, 0
#                 states, actions, rewards, values, dones, logps, ents = [], [], [], [], [], [], []

#                 while (not done) and step < MAX_STEPS_PER_EPISODE and state[0] < env[arm_name].maxWait:
#                     step += 1
#                     s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

#                     with torch.no_grad():
#                         epsilon = 0.2
#                         idx = ac.act(s_t)
#                         logits = idx - lam_tensor
#                         dist = torch.distributions.Bernoulli(logits=logits)
#                         if torch.rand(1).item() < epsilon:
#                             a_t = torch.bernoulli(torch.full_like(logits, 0.5))
#                             #a_t = torch.bernoulli(torch.full_like(logits, 0))
#                         else:
#                             a_t = dist.sample()
#                         a_t = a_t.view(1, 1)
#                         logp = dist.log_prob(a_t)
#                         ent = dist.entropy()
#                         v_t = ac.value(s_t, lam_tensor)

#                     reward = env[arm_name]._calReward(int(a_t.item()), state[0])
#                     if a_t.item() == 1:
#                         reward -= lam_val
#                     next_state, _, _, _ = env[arm_name].step(int(a_t.item()))
#                     states.append(s_t)
#                     actions.append(a_t)
#                     rewards.append(reward)
#                     values.append(v_t.item())
#                     logps.append(logp)
#                     ents.append(ent)
#                     dones.append(done)
#                     state = next_state
#                     if state[0] == 1:
#                         break

#                 next_v = ac.value(
#                     torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
#                     lam_tensor
#                 ).item()
#                 returns = compute_gae(rewards, values, next_v, dones)
#                 adv = torch.tensor(returns, device=device) - torch.tensor(values, device=device)
#                 all_s += states
#                 all_a += actions
#                 all_r += returns
#                 all_v += values
#                 all_logp += logps
#                 all_ent += ents
#                 all_lam += [lam_val] * len(states)
#                 all_adv += adv.tolist()

#             S = torch.cat(all_s)
#             A = torch.stack(all_a)
#             RET = torch.tensor(all_r, dtype=torch.float32, device=device).unsqueeze(1)
#             V = torch.tensor(all_v, dtype=torch.float32, device=device).unsqueeze(1)
#             ADV = torch.tensor(all_adv, dtype=torch.float32, device=device).unsqueeze(1)
#             ADV = (ADV - ADV.mean()) / (ADV.std(unbiased=False) + 1e-8)
#             LAM = torch.tensor(all_lam, dtype=torch.float32, device=device).unsqueeze(1)

#             for _ in range(N_CRITIC_UPDATES):
#                 critic_opt.zero_grad()
#                 v_pred = ac.value(S, LAM)
#                 v_loss = ((v_pred - RET) ** 2).mean()
#                 v_loss.backward()
#                 nn.utils.clip_grad_norm_(ac.critic.parameters(), GRAD_CLIP_NORM)
#                 critic_opt.step()

#             with torch.no_grad():
#                 S_np = S.cpu().numpy()
#                 LAM_np = LAM.cpu().numpy()
#                 reward1, reward0 = [], []
#                 for i in range(len(S_np)):
#                     state_i = S_np[i][0]
#                     lam_i = LAM_np[i][0]
#                     r1 = env[arm_name]._calReward(1, state_i) - lam_i
#                     r0 = env[arm_name]._calReward(0, state_i)
#                     ns1 = np.array([1.0], dtype=np.float32)
#                     ns0 = np.array([min(state_i + 1, env[arm_name].maxWait)], dtype=np.float32)
#                     s1 = torch.tensor(ns1, dtype=torch.float32, device=device).unsqueeze(0)
#                     s0 = torch.tensor(ns0, dtype=torch.float32, device=device).unsqueeze(0)
#                     v1 = r1 + GAMMA * ac.value(s1, torch.tensor([[lam_i]], dtype=torch.float32, device=device)).item()
#                     v0 = r0 + GAMMA * ac.value(s0, torch.tensor([[lam_i]], dtype=torch.float32, device=device)).item()
#                     reward1.append(v1)
#                     reward0.append(v0)
#                 target = (np.array(reward1) > np.array(reward0)).astype(np.float32)
#                 target = torch.tensor(target, device=device).unsqueeze(1)

#             actor_opt.zero_grad()
#             idx_pred = ac.act(S)
#             logits = idx_pred - LAM
#             bce_loss = nn.BCEWithLogitsLoss()(logits, target)
#             prob = torch.sigmoid(logits)
#             entropy = - (prob * torch.log(prob + 1e-8) + (1 - prob) * torch.log(1 - prob + 1e-8))
#             loss = bce_loss - ENTROPY_COEF * entropy.mean()
#             loss.backward()
#             nn.utils.clip_grad_norm_(ac.actor.parameters(), GRAD_CLIP_NORM)
#             actor_opt.step()

#             tqdm_bar.set_description(
#                 f"{arm_name} | Ep {ep + 1}/{TOTAL_EPISODES} | Aloss {loss.item():.4f} | Vloss {v_loss.item():.4f}"
#             )

#             loss_log.append(loss.item())
#             vloss_log.append(v_loss.item())

#         torch.save(ac.state_dict(), os.path.join(subdir, "actor_critic.pth"))
#         np.save(os.path.join(subdir, "actor_loss.npy"), np.array(loss_log))
#         np.save(os.path.join(subdir, "critic_loss.npy"), np.array(vloss_log))

#         with torch.no_grad():
#             s_states = torch.arange(1, env[arm_name].maxWait + 1, dtype=torch.float32).unsqueeze(1).to(device)
#             whittle_index = ac.actor(s_states).cpu().numpy()
#             np.save(os.path.join(subdir, "whittle_index.npy"), whittle_index)


# if __name__ == "__main__":
#     train_multi_arm("10arms", get_10_arm_env)
#     train_multi_arm("20arms", get_20_arm_env)
