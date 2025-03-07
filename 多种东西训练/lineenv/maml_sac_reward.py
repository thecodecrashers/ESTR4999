import torch
import numpy as np
from lineEnv import lineEnv
import matplotlib.pyplot as plt
from tqdm import tqdm
from maml_sac import Actor, Critic, clone_model, compute_sac_losses, ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置
N = 10                # 臂的数量
M = 3                   # 每个时刻最多启动的臂数量
OptX = 99
state_dim = 1
env_features_dim = 4
inner_lr = 0.01
adaptation_steps = 128
batch_size = 64
gamma = 0.99
alpha = 0.2
memory_size = 10000

# 加载Meta模型参数
meta_actor = Actor(state_dim, env_features_dim).to(device)
meta_actor.load_state_dict(torch.load("maml_sac/meta_actor.pth", map_location=device))
meta_actor.eval()

meta_critic = Critic(state_dim, env_features_dim).to(device)
meta_critic.load_state_dict(torch.load("maml_sac/meta_critic.pth", map_location=device))
meta_critic.eval()

# 初始化臂的环境
pq_pairs = [(np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)) for _ in range(N)]

arms_env = []
whittle_indices = []

for idx, (p, q) in enumerate(tqdm(pq_pairs, desc="Calculating Whittle Indices")):
    lam = np.random.uniform(0, 5)
    env = lineEnv(seed=idx, N=N, OptX=OptX, p=p, q=q)
    arms_env.append(env)
    env_embed = torch.tensor([p, q, float(OptX), lam], dtype=torch.float32, device=device)

    # Clone 并快速微调 Actor 与 Critic
    actor_fast = clone_model(meta_actor)
    critic_fast = clone_model(meta_critic)
    actor_optim = torch.optim.SGD(actor_fast.parameters(), lr=inner_lr)
    critic_optim = torch.optim.SGD(critic_fast.parameters(), lr=inner_lr)

    buffer = ReplayBuffer(max_size=memory_size)

    state = env.reset()
    for _ in range(adaptation_steps):
        state_t = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            logit = actor_fast(state_t, env_embed)
            p1 = torch.sigmoid(logit - lam)
            action = int((p1 > 0.5).item())

        next_state, reward, done, _ = env.step(action)
        reward_adj = reward - lam if action == 1 else reward
        buffer.push(state, action, reward_adj, next_state, env_embed.cpu().numpy())

        state = next_state if not done else env.reset()

        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            actor_loss, critic_loss = compute_sac_losses(
                actor_fast, critic_fast, batch, gamma, alpha, device
            )

            actor_optim.zero_grad()
            critic_optim.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            actor_optim.step()
            critic_optim.step()

    # 微调后 Actor 网络的输出作为 Whittle Index
    with torch.no_grad():
        whittle_index = actor_fast(torch.tensor([0], dtype=torch.float32, device=device), env_embed).item()
        whittle_indices.append(whittle_index)


print(whittle_indices)
# 长期回报评估
episodes = 500
steps_per_episode = 100
cumulative_rewards = []

states = [env.reset() for env in arms_env]

for _ in tqdm(range(episodes), desc="Evaluating Long-term Reward"):
    total_reward = 0
    for _ in range(steps_per_episode):
        selected_indices = np.argsort(whittle_indices)[-M:]
        step_reward = 0
        for idx, env in enumerate(arms_env):
            action = 1 if idx in selected_indices else 0
            next_state, reward, done, _ = env.step(action)
            step_reward += reward
            states[idx] = next_state if not done else env.reset()
        total_reward += step_reward
    cumulative_rewards.append(total_reward / steps_per_episode)

average_reward = np.mean(cumulative_rewards)
print(f"平均长期回报: {average_reward:.4f}")

# 绘制Whittle Index分布
plt.figure(figsize=(10, 6))
plt.bar(range(N), whittle_indices)
plt.xlabel('Arm Index')
plt.ylabel('Whittle Index')
plt.title('Whittle Index for Each Arm')
plt.grid(True)
plt.savefig("whittle_indices.png")
plt.show()

# 绘制长期回报曲线
plt.figure(figsize=(10, 6))
plt.plot(cumulative_rewards, label='Average Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.savefig("long_term_rewards.png")
plt.show()
