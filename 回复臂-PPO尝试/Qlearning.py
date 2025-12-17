import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from rc_env import recoveringBanditsEnv
from model import ActorCritic
from tqdm import tqdm

# ========= 设备选择 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= 超参数 =========
GAMMA = 0.99
GAE_LAMBDA = 0.95
TOTAL_EPISODES = 10000#设置成十万次是可以学出来需要的效果的我是真服了，一万次也可以
EPOCHS = 6
BATCH_SIZE = 256
LAMBDA_MIN = 1
LAMBDA_MAX = 11
NUM_LAMBDAS = 10
MAX_STEPS_PER_EPISODE = 400
WARMUP_RATIO = 1 # 前30% episodes 用 BCE warmup

# 学习率和更新频率
ACTOR_LR = 1e-3
CRITIC_LR = 1e-4
N_CRITIC_UPDATES = 3
ENTROPY_COEF = 0.01

def visualize_index(ac, max_state=40):
    ac.actor.eval()
    states = torch.arange(1, max_state + 1, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        index_preds = ac.actor(states)
    states_np = states.cpu().numpy().squeeze()
    index_np = index_preds.cpu().numpy().squeeze()
    plt.figure(figsize=(8, 5))
    plt.plot(states_np, index_np, label="Whittle Index Estimate", marker='o')
    plt.xlabel("State")
    plt.ylabel("Predicted Index")
    plt.title("Actor Network: Whittle Index vs. State")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_gae(rewards, values, next_value, dones):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])
    return returns

def train():
    thetaVals = [10, 0.5, 0.0]
    max_wait = 100
    noise_var = 0
    seed = 42
    env = recoveringBanditsEnv(seed=seed, thetaVals=thetaVals, noiseVar=noise_var, maxWait=max_wait)

    state_dim = 1
    ac = ActorCritic(state_dim).to(device)

    actor_optimizer = optim.Adam(ac.actor.parameters(), lr=ACTOR_LR)
    critic_optimizer = optim.Adam(ac.critic.parameters(), lr=CRITIC_LR)

    lambda_bins = np.linspace(LAMBDA_MIN, LAMBDA_MAX, NUM_LAMBDAS + 1)
    warmup_episode = int(TOTAL_EPISODES * WARMUP_RATIO)
    tqdm_bar = tqdm(range(TOTAL_EPISODES), desc="Training Progress")
    for episode in tqdm_bar:
        sampled_lambdas = [np.random.uniform(lambda_bins[i], lambda_bins[i+1]) for i in range(NUM_LAMBDAS)]
        all_states, all_actions, all_returns, all_values, all_lambdas, all_log_probs, all_entropies = [], [], [], [], [], [], []

        for lambda_val in sampled_lambdas:
            lambda_tensor = torch.tensor([[lambda_val]], dtype=torch.float32, device=device)
            state = env.reset()
            env.arm[0]=np.random.randint(1, max_wait)
            state = np.array([env.arm[0]], dtype=np.float32)
            done = False
            step = 0
            states, actions, rewards, values, log_probs, entropies, dones = [], [], [], [], [], [], []

            while not done and step < MAX_STEPS_PER_EPISODE and state[0] < max_wait:
                step += 1
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                # whittle_index = ac.act(state_tensor)
                # prob = torch.sigmoid(whittle_index - lambda_tensor)
                # dist = torch.distributions.Bernoulli(prob)
                # action = dist.sample()
                # log_prob = dist.log_prob(action)
                # entropy = dist.entropy()
                with torch.no_grad():
                    # 当前状态
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    # 尝试 action = 0
                    reward_0 = env._calReward(0, state[0])
                    state_after_0 = min(state[0] + 1, max_wait)
                    next_tensor_0 = torch.tensor([[state_after_0]], dtype=torch.float32, device=device)
                    v_0 = reward_0 + GAMMA * ac.value(next_tensor_0, lambda_tensor)

                    # 尝试 action = 1
                    reward_1 = env._calReward(1, state[0]) - lambda_val
                    state_after_1 = 1
                    next_tensor_1 = torch.tensor([[state_after_1]], dtype=torch.float32, device=device)
                    v_1 = reward_1 + GAMMA * ac.value(next_tensor_1, lambda_tensor)
                    #print("v_0:", v_0.item(), "v_1:", v_1.item(), "lambda:", lambda_val)
                    # 贪婪决策
                    #action = torch.tensor([[1]]) if v_1 > v_0 else torch.tensor([[0]])
                    epsilon = 0.2  # 你可以把它设为一个随 episode 衰减的超参数
                    if torch.rand(1).item() < epsilon:
                        # 探索：随机选一个动作
                        action = torch.randint(0, 2, (1, 1), dtype=torch.float32, device=device)
                    else:
                        # 利用：选择 v 更大的动作
                        action = torch.tensor([[1.0]], device=device) if v_1 > v_0 else torch.tensor([[0.0]], device=device)


                value = ac.value(state_tensor, lambda_tensor)
                reward = env._calReward(int(action.item()), state[0])
                if action.item() == 1:
                    reward -= lambda_val

                next_state, _, _, _ = env.step(int(action.item()))

                states.append(state_tensor)
                actions.append(action)
                rewards.append(reward)
                values.append(value.item())
                dones.append(done)

                state = next_state
                if(state[0]==1):
                    break
                #print("step:", step, "action:", action.item(), "next_state:", next_state)

            next_state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            next_value = ac.value(next_state_tensor, lambda_tensor).item()

            returns = compute_gae(rewards, values, next_value, dones)
            advantages = torch.tensor(returns, dtype=torch.float32, device=device) - \
                         torch.tensor(values, dtype=torch.float32, device=device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            all_states.extend(states)
            all_actions.extend(actions)
            all_returns.extend(returns)
            all_values.extend(advantages)
            all_lambdas.extend([lambda_val] * len(states))

        # 合并 batch 数据
        states = torch.cat(all_states).to(device)
        actions = torch.stack(all_actions).to(device)
        returns = torch.tensor(all_returns, dtype=torch.float32, device=device).unsqueeze(1)
        advantages = torch.tensor(all_values, dtype=torch.float32, device=device).unsqueeze(1)
        lambdas_tensor = torch.tensor(all_lambdas, dtype=torch.float32, device=device).unsqueeze(1)

        for epoch in range(EPOCHS):
            for i in range(0, len(states), BATCH_SIZE):
                s_batch = states[i:i+BATCH_SIZE]
                ret_batch = returns[i:i+BATCH_SIZE]
                lam_batch = lambdas_tensor[i:i+BATCH_SIZE]


                # 更新 critic
                critic_optimizer.zero_grad()
                value_pred = ac.value(s_batch, lam_batch)
                value_loss = ((value_pred - ret_batch) ** 2).mean()
                value_loss.backward()
                critic_optimizer.step()
                
        if(episode% N_CRITIC_UPDATES == 0):        
            for i in range(0, len(states), BATCH_SIZE):
                s_batch = states[i:i+BATCH_SIZE]
                lam_batch = lambdas_tensor[i:i+BATCH_SIZE]

                with torch.no_grad():
                    # === Critic 判断 action=0 和 action=1 的 Q 值 ===
                    reward_0 = torch.tensor(
                        [env._calReward(0, s.item()) for s in s_batch], dtype=torch.float32, device=device
                    ).unsqueeze(1)
                    next_state_0 = torch.tensor(
                        [min(s.item() + 1, env.maxWait) for s in s_batch], dtype=torch.float32, device=device
                    ).unsqueeze(1)
                    v_next_0 = ac.value(next_state_0, lam_batch)
                    R0 = reward_0 + GAMMA * v_next_0

                    reward_1 = torch.tensor(
                        [env._calReward(1, s.item()) - lam.item() for s, lam in zip(s_batch, lam_batch)],
                        dtype=torch.float32, device=device
                    ).unsqueeze(1)
                    next_state_1 = torch.tensor(
                        [1.0 for _ in s_batch], dtype=torch.float32, device=device
                    ).unsqueeze(1)  # action=1 后 always 回到 state=1
                    v_next_1 = ac.value(next_state_1, lam_batch)
                    R1 = reward_1 + GAMMA * v_next_1

                    # 最优动作作为 BC 监督信号
                    target_action = (R1 > R0).float()  # shape: (B, 1)

                # === Actor 输出概率 ===
                actor_optimizer.zero_grad()
                pred_index = ac.act(s_batch)  # shape: (B, 1)
                pred_prob = torch.sigmoid(10*(pred_index - lam_batch))
                actor_loss = nn.BCELoss()(pred_prob, target_action)
                actor_loss.backward()
                actor_optimizer.step()
# === Actor 更新 ===
                """if (epoch * (len(states) // BATCH_SIZE) + i // BATCH_SIZE) % N_CRITIC_UPDATES == 0:
                    actor_optimizer.zero_grad()

                    # === 1. 当前状态与 lambda ===
                    s_batch = states[i:i+BATCH_SIZE]
                    lam_batch = lambdas_tensor[i:i+BATCH_SIZE]

                    with torch.no_grad():
                        # === 2. 使用 Critic 来评估动作 0 和 1 的回报 ===
                        reward_0 = torch.tensor(
                            [env._calReward(0, s.item()) for s in s_batch], dtype=torch.float32, device=device
                        ).unsqueeze(1)
                        next_state_0 = torch.tensor(
                            [env.step(0)[0][0] if env.arm.update({0: int(s.item())}) is None else 0 for s in s_batch],
                            dtype=torch.float32, device=device
                        ).unsqueeze(1)
                        v_next_0 = ac.value(next_state_0, lam_batch)
                        R0 = reward_0 + GAMMA * v_next_0

                        reward_1 = torch.tensor(
                            [env._calReward(1, s.item()) - lam.item() for s, lam in zip(s_batch, lam_batch)],
                            dtype=torch.float32, device=device
                        ).unsqueeze(1)
                        next_state_1 = torch.tensor(
                            [env.step(1)[0][0] if env.arm.update({0: int(s.item())}) is None else 0 for s in s_batch],
                            dtype=torch.float32, device=device
                        ).unsqueeze(1)
                        v_next_1 = ac.value(next_state_1, lam_batch)
                        R1 = reward_1 + GAMMA * v_next_1
                        # === 3. 谁回报大，就设为“目标动作” ===
                        target_action = (R1 > R0).float()  # 形状：(B, 1)

                    # === 4. Actor 输出当前动作概率 ===
                    pred_index = ac.act(s_batch)  # shape: (B, 1)
                    pred_prob = torch.sigmoid(pred_index - lam_batch)  # shape: (B, 1)

                    # === 5. 用 BCE Loss 让 actor 输出接近 critic 给出的最优动作 ===
                    actor_loss = nn.BCELoss()(pred_prob, target_action)

                    actor_loss.backward()
                    actor_optimizer.step()"""
       # === 标准 Actor-Critic Actor 更新 ===
# 每 N 次批次更新一次 actor
                """if (epoch * (len(states) // BATCH_SIZE) + i // BATCH_SIZE) % N_CRITIC_UPDATES == 0:
                    actor_optimizer.zero_grad()

                    s_batch = states[i:i+BATCH_SIZE]
                    lam_batch = lambdas_tensor[i:i+BATCH_SIZE]

                    target_action_list = []

                    for s_tensor, lam in zip(s_batch, lam_batch):
                        s_val = s_tensor.item()
                        lambda_val = lam.item()

                        # === 保存当前环境状态 ===
                        env_state_backup = env.arm[0]

                        # === 估计 Action = 0 的 R0 ===
                        env.arm[0] = int(s_val)
                        next_state_0, reward_0, _, _ = env.step(0)
                        s_next_0_tensor = torch.tensor(next_state_0, dtype=torch.float32, device=device).unsqueeze(0)
                        v_next_0 = ac.critic(s_next_0_tensor, lam.unsqueeze(0))
                        R0 = reward_0 + GAMMA * v_next_0.item()

                        # === 估计 Action = 1 的 R1 ===
                        env.arm[0] = int(s_val)
                        next_state_1, reward_1, _, _ = env.step(1)
                        reward_1 -= lambda_val
                        s_next_1_tensor = torch.tensor(next_state_1, dtype=torch.float32, device=device).unsqueeze(0)
                        v_next_1 = ac.critic(s_next_1_tensor, lam.unsqueeze(0))
                        R1 = reward_1 + GAMMA * v_next_1.item()

                        # === 比较哪个好 ===
                        best_action = 1.0 if R1 > R0 else 0.0
                        target_action_list.append(best_action)

                        # === 恢复环境状态 ===
                        env.arm[0] = env_state_backup

                    # === 构建目标动作张量 ===
                    target_action = torch.tensor(target_action_list, dtype=torch.float32, device=device).unsqueeze(1)

                    # === 预测当前 actor 输出 ===
                    pred_index = ac.act(s_batch)
                    pred_prob = torch.sigmoid(pred_index - lam_batch)

                    # === 用 BCE 训练 actor 靠近 “最佳动作” ===
                    actor_loss = nn.BCELoss()(pred_prob, target_action)

                    actor_loss.backward()
                    actor_optimizer.step()
"""

                """if (epoch * (len(states) // BATCH_SIZE) + i // BATCH_SIZE) % N_CRITIC_UPDATES == 0:
                    actor_optimizer.zero_grad()

                    # 用 actor 当前策略重新预测动作分布
                    pred_index = ac.act(s_batch)
                    prob = torch.sigmoid(pred_index - lam_batch)
                    dist = torch.distributions.Bernoulli(prob)

                    # 计算当前策略下的 log_prob
                    logp = dist.log_prob(a_batch)

                    # === 使用 Critic 提供的 advantage 来指导策略优化 ===
                    actor_loss = -(logp * adv_batch).mean() - ENTROPY_COEF * dist.entropy().mean()

                    actor_loss.backward()
                    actor_optimizer.step()"""

                """# 更新 actor
                if (epoch * (len(states) // BATCH_SIZE) + i // BATCH_SIZE) % N_CRITIC_UPDATES == 0:
                    actor_optimizer.zero_grad()
                    pred_index = ac.act(s_batch)
                    prob = torch.sigmoid(pred_index - lam_batch)
                    dist = torch.distributions.Bernoulli(prob)
                    logp_new = dist.log_prob(a_batch)

                    if episode < warmup_episode:
                        a_batch = a_batch.squeeze(-1)  # (B, 1, 1) → (B, 1)
                        actor_loss = nn.BCELoss()(prob, a_batch)
                    else:
                        ratio = torch.exp(logp_new - logp_old)
                        surr1 = ratio * adv_batch
                        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv_batch
                        actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy_batch.mean()

                    actor_loss.backward()
                    actor_optimizer.step()"""
        tqdm_bar.set_description(f"Ep {episode+1}/{TOTAL_EPISODES} | ActorLoss: {actor_loss:.4f} | ValueLoss: {value_loss:.4f}")

    return ac

ac_model = train()
visualize_index(ac_model)




"""import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rc_env import recoveringBanditsEnv
from model import ActorCritic
import matplotlib.pyplot as plt

# ========= 设备选择 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= 超参数 =========
GAMMA = 0.99
GAE_LAMBDA = 0.95
TOTAL_EPISODES = 500
EPOCHS = 5
BATCH_SIZE = 64
LAMBDA_MIN = 0.01
LAMBDA_MAX = 1.0
NUM_LAMBDAS = 5
MAX_STEPS_PER_EPISODE = 200

# 学习率和更新频率
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
N_CRITIC_UPDATES = 3  # 每3次Critic更新才更新1次Actor

def visualize_index(ac, max_state=40):
    ac.actor.eval()  # 只用 actor 网络，不涉及 critic 和 lambda

    # 构造状态输入：1 到 max_state
    states = torch.arange(1, max_state + 1, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        index_preds = ac.actor(states)

    states_np = states.cpu().numpy().squeeze()
    index_np = index_preds.cpu().numpy().squeeze()

    # 画图
    plt.figure(figsize=(8, 5))
    plt.plot(states_np, index_np, label="Whittle Index Estimate", marker='o')
    plt.xlabel("State")
    plt.ylabel("Predicted Index")
    plt.title("Actor Network: Whittle Index vs. State")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_gae(rewards, values, next_value, dones):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])
    return returns

def train():
    thetaVals = [0.5, 0.5, 0.0]
    max_wait = 50
    noise_var = 0.01
    seed = 42
    env = recoveringBanditsEnv(seed=seed, thetaVals=thetaVals, noiseVar=noise_var, maxWait=max_wait)

    state_dim = env.observation_space.shape[0]
    ac = ActorCritic(state_dim).to(device)

    actor_optimizer = optim.Adam(ac.actor.parameters(), lr=ACTOR_LR)
    critic_optimizer = optim.Adam(ac.critic.parameters(), lr=CRITIC_LR)

    lambda_bins = np.linspace(LAMBDA_MIN, LAMBDA_MAX, NUM_LAMBDAS + 1)

    for episode in range(TOTAL_EPISODES):
        sampled_lambdas = [
            np.random.uniform(lambda_bins[i], lambda_bins[i+1])
            for i in range(NUM_LAMBDAS)
        ]

        all_states, all_actions, all_returns, all_values, all_lambdas = [], [], [], [], []

        for lambda_val in sampled_lambdas:
            lambda_tensor = torch.tensor([[lambda_val]], dtype=torch.float32, device=device)
            state = env.reset()
            done = False
            step = 0
            states, actions, rewards, values, dones = [], [], [], [], []

            while not done and step < MAX_STEPS_PER_EPISODE:
                step += 1
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    whittle_index = ac.act(state_tensor)
                    prob = torch.sigmoid(whittle_index - lambda_tensor)
                    action = int(prob.item() > 0.5)
                    value = ac.value(state_tensor, lambda_tensor)

                reward = env._calReward(action, state[0])
                if action == 1:
                    reward -= lambda_val

                next_state, _, _, _ = env.step(action)

                states.append(state_tensor)
                actions.append(torch.tensor([[action]], dtype=torch.float32, device=device))
                rewards.append(reward)
                values.append(value.item())
                dones.append(done)

                state = next_state

            with torch.no_grad():
                next_state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                next_value = ac.value(next_state_tensor, lambda_tensor).item()

            returns = compute_gae(rewards, values, next_value, dones)
            advantages = torch.tensor(returns, dtype=torch.float32, device=device) - \
                         torch.tensor(values, dtype=torch.float32, device=device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            all_states.extend(states)
            all_actions.extend(actions)
            all_returns.extend(returns)
            all_values.extend(advantages)
            all_lambdas.extend([lambda_val] * len(states))

        # ===== Prepare batch tensors =====
        states = torch.cat(all_states).to(device)
        actions = torch.cat(all_actions).to(device)
        returns = torch.tensor(all_returns, dtype=torch.float32, device=device).unsqueeze(1)
        advantages = torch.tensor(all_values, dtype=torch.float32, device=device).unsqueeze(1)
        lambdas_tensor = torch.tensor(all_lambdas, dtype=torch.float32, device=device).unsqueeze(1)

        # ======= 分开优化：Critic 优先 =======
        for epoch in range(EPOCHS):
            for i in range(0, len(states), BATCH_SIZE):
                s_batch = states[i:i+BATCH_SIZE]
                a_batch = actions[i:i+BATCH_SIZE]
                ret_batch = returns[i:i+BATCH_SIZE]
                adv_batch = advantages[i:i+BATCH_SIZE]
                lam_batch = lambdas_tensor[i:i+BATCH_SIZE]

                # ---- Critic update ----
                critic_optimizer.zero_grad()
                value_pred = ac.value(s_batch, lam_batch)
                value_loss = ((value_pred - ret_batch) ** 2).mean()
                value_loss.backward()
                critic_optimizer.step()

                # 每 N 次 critic 更新，更新一次 actor
                if (epoch * (len(states) // BATCH_SIZE) + i // BATCH_SIZE) % N_CRITIC_UPDATES == 0:
                    actor_optimizer.zero_grad()
                    pred_index = ac.act(s_batch)
                    prob = torch.sigmoid(pred_index - lam_batch)
                    pred_action = prob
                    actor_loss = nn.BCELoss()(pred_action, a_batch)
                    
                    actor_loss.backward()
                    actor_optimizer.step()
    return ac

if __name__ == "__main__":
    ac=train()
    visualize_index(ac)"""

