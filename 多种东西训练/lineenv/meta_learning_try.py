import torch
import torch.nn as nn
import numpy as np
import random
import lineEnv  # 你的自定义环境
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

# 自动检测 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# 超参数
# ------------------------------
state_dim = 1
action_dim = 1  # 输出一个连续值，然后再用 sigmoid->bernoulli 得到 0/1 动作
env_features_dim = 3
embed_dim = 16
nb_arms = 15

gamma = 0.99
batch_size = 16
memory_size = 1000
# 下面这两个是 TD3 常用的策略：Critic 每一步都更新，而 Actor 和目标网络延迟更新
policy_update_delay = 2  # 每隔多少次更新 Critic 之后才更新 Actor
tau = 0.005  # 软更新系数(可改大一点也行，这里示例)

num_episodes = 10
max_steps_per_episode = 20

# 生成 (p, q) 组合
prob_values = np.linspace(start=0.1, stop=0.9, num=nb_arms)
pq_selected = [(p, q) for p, q in zip(prob_values, reversed(prob_values))]

m = 10  # 用于计算动作概率：sigmoid(m*(actor_out - lambda_val))

learning_rate = 1e-3

# ------------------------------
# 经验回放缓冲区
# ------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        存储 (state, action, reward, next_state, done)
        都先存为标量或一维即可，在采样时再转成张量。
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        # 转换为 (batch_size, 1) 张量
        return (
            torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(-1)
        )

    def __len__(self):
        return len(self.buffer)

# ------------------------------
# 网络定义
# ------------------------------
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmbeddingNet, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class Actor(nn.Module):
    """
    Actor 输出一个(批大小, 1)的值，后面会再做 “baseline(lambda_val)差分 + sigmoid + bernoulli” 得到离散动作。
    """
    def __init__(self, state_dim, env_embed_dim, action_dim):
        super(Actor, self).__init__()
        self.input_dim = state_dim + env_embed_dim
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)  # 这里是 1
        )

    def forward(self, state, env_embed):
        # state: (batch_size, state_dim)
        # env_embed: (batch_size, env_embed_dim)
        combined_input = torch.cat([state, env_embed], dim=-1)
        return self.model(combined_input)


class Critic(nn.Module):
    """
    这里的 Critic 是单路输出 Q(state, action)。
    """
    def __init__(self, state_dim, action_dim, env_embed_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + env_embed_dim, 128)
        self.fc2 = nn.Linear(128 + action_dim, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action, env_embed):
        # state, action, env_embed 形状均为 (batch_size, 1/嵌入维度)
        x = torch.cat([state, env_embed], dim=-1)
        x = self.relu(self.fc1(x))
        x = torch.cat([x, action], dim=-1)
        x = self.relu(self.fc2(x))
        q = self.fc3(x)
        return q


def soft_update(source_net, target_net, tau=0.005):
    """
    软更新：target_param = tau * source_param + (1 - tau)* target_param
    """
    for tp, sp in zip(target_net.parameters(), source_net.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# ------------------------------
# 主训练循环
# ------------------------------

all_critic_losses = []
all_actor_losses = []

for i, (p, q) in enumerate(pq_selected):
    print(f"\n========== 训练臂 {i}: p={p:.2f}, q={q:.2f} ==========")

    # 创建环境
    env = lineEnv.lineEnv(seed=42, N=10, OptX=99, p=p, q=q)

    # 初始化各网络
    embedding_net = EmbeddingNet(env_features_dim, embed_dim).to(device)

    actor = Actor(state_dim, embed_dim, action_dim).to(device)
    critic1 = Critic(state_dim, action_dim, embed_dim).to(device)
    critic2 = Critic(state_dim, action_dim, embed_dim).to(device)

    target_actor = Actor(state_dim, embed_dim, action_dim).to(device)
    target_critic1 = Critic(state_dim, action_dim, embed_dim).to(device)
    target_critic2 = Critic(state_dim, action_dim, embed_dim).to(device)

    # 初始化目标网络参数与主网络相同
    target_actor.load_state_dict(actor.state_dict())
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    # 优化器
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    critic1_optimizer = torch.optim.Adam(critic1.parameters(), lr=learning_rate)
    critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=learning_rate)

    # 经验回放
    memory = ReplayBuffer(memory_size)

    # 记录损失
    critic_losses = []
    actor_losses = []

    # 训练过程中，为了保证每个 episode 都有新的交互，直接在循环中每个 step 都与环境交互
    # 再将数据存入 buffer，并做一次更新（若 buffer 足够大）。
    total_steps = 0  # 用来判断什么时候延迟更新

    # 生成随机的环境特征，并嵌入
    # 如果每个episode都变的话，就每次episode都写在里面；也可以只在外面生成一次
    env_features = torch.tensor([99, p, q], dtype=torch.float32)  # 确保是Tensor
    env_embed_full = embedding_net(env_features).detach()  # (1, embed_dim)
    env_embed_full = env_embed_full # 确保批量维度正确
    env_embed_full = env_embed_full.view(1, 1, -1)  # 变成 (1, 1, 16)

    pbar = tqdm(range(num_episodes), desc=f"Training Arm {i}")
    for episode in pbar:
        lambda_val_target = np.random.uniform(-100, 100)
        state_val = env.reset()  # 重置环境，返回 int
        # 转成tensor (1, 1)
        state = torch.tensor([[state_val]], dtype=torch.float32, device=device)

        episode_reward = 0.0

        for step in range(max_steps_per_episode):
            total_steps += 1

            # ---------------- 1) 与环境交互 ----------------
            # （先从当前 state 算出 Actor 输出与 baseline）
            with torch.no_grad():
                actor_out = actor(state, env_embed_full)
                # 相当于 baseline
                lambda_val = actor(state, env_embed_full)  # 这里可以用同一个，也可以额外训练一个 baseline
                # 动作概率
                action_prob = torch.sigmoid(m * (actor_out - lambda_val))
                # 采样离散动作
                action_sample = torch.bernoulli(action_prob).detach()
            
            action_int = int(action_sample.item())

            # 与环境交互
            next_state_val, reward, done, _ = env.step(action_int)

            # 存入 ReplayBuffer
            memory.push(state_val, action_int, reward, next_state_val, done)

            # 转到下一个state
            state_val = next_state_val
            state = torch.tensor([[state_val]], dtype=torch.float32, device=device)
            episode_reward += reward

            # ---------------- 2) 从 ReplayBuffer 采样并更新网络 ----------------
            if len(memory) >= batch_size:
                # 取批量
                states_b, actions_b, rewards_b, next_states_b, dones_b = memory.sample(batch_size)
                # env_embed 批量复制
                env_embed_b = env_embed_full
                print(env_embed_b)
                print(next_states_b)
                # ==== 2.1 更新 Critic1、Critic2 ====
                with torch.no_grad():
                    next_actor_out = target_actor(next_states_b, env_embed_b)
                    next_action_prob = torch.sigmoid(m * (next_actor_out - lambda_val_target))
                    # 这里是离散 Bernoulli 的采样
                    next_action_sample = torch.bernoulli(next_action_prob)

                    # 目标 Q(s', a')
                    target_q1 = target_critic1(next_states_b, next_action_sample, env_embed_b)
                    target_q2 = target_critic2(next_states_b, next_action_sample, env_embed_b)
                    target_q_min = torch.min(target_q1, target_q2)
                    target_q = rewards_b + (1.0 - dones_b) * gamma * target_q_min

                # 当前 Q(s, a)
                current_q1 = critic1(states_b, actions_b, env_embed_b)
                current_q2 = critic2(states_b, actions_b, env_embed_b)

                critic1_loss = nn.MSELoss()(current_q1, target_q)
                critic2_loss = nn.MSELoss()(current_q2, target_q)
                critic_loss = critic1_loss + critic2_loss

                critic1_optimizer.zero_grad()
                critic2_optimizer.zero_grad()
                critic_loss.backward()
                critic1_optimizer.step()
                critic2_optimizer.step()

                critic_losses.append(critic_loss.item())

                # ==== 2.2 延迟更新 Actor (和目标网络) ====
                if total_steps % policy_update_delay == 0:
                    # Actor 的 loss = - E[ Q1(s, \pi(s)) ]
                    # 注意这里要跟 Actor 的定义对应（需要先计算 baseline，然后再用 Actor out - baseline -> sigmoid -> Bernoulli）
                    # 但是对离散动作直接做近似梯度很怪，这里只是示例
                    actor_out_pred = actor(states_b, env_embed_b)
                    lambda_val_pred = actor(states_b, env_embed_b)
                    action_prob_pred = torch.sigmoid(m * (actor_out_pred - lambda_val_pred))
                    # 用其期望(注意这是Bernoulli分布, 期望 = action_prob_pred) 来近似做 Actor 更新
                    # 也可直接用硬采样 + reinforce，但那就需要策略梯度。这里只是简单做个近似。
                    # 将“期望动作”当做连续值输入Critic
                    # Critic 这里也接收的是离散 action，但我们暂且把 action_prob_pred 当做“伪连续值”输入
                    q1_of_actor = critic1(states_b, action_prob_pred, env_embed_b)
                    actor_loss = -q1_of_actor.mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    actor_losses.append(actor_loss.item())

                    # 软更新目标网络
                    soft_update(actor, target_actor, tau)
                    soft_update(critic1, target_critic1, tau)
                    soft_update(critic2, target_critic2, tau)

            if done:
                break

        pbar.set_postfix({
            "EpisodeReward": f"{episode_reward:.2f}",
            "CriticLoss": f"{critic_losses[-1] if len(critic_losses)>0 else 0:.4f}",
            "ActorLoss": f"{actor_losses[-1] if len(actor_losses)>0 else 0:.4f}"
        })

    # 记录该臂的损失
    all_critic_losses.extend(critic_losses)
    all_actor_losses.extend(actor_losses)

    # ========== 将该臂的网络保存到本地 ==========
    torch.save(embedding_net.state_dict(), f"embedding_net_arm{i}.pth")
    torch.save(actor.state_dict(), f"actor_arm{i}.pth")
    torch.save(critic1.state_dict(), f"critic1_arm{i}.pth")
    torch.save(critic2.state_dict(), f"critic2_arm{i}.pth")
    torch.save(target_actor.state_dict(), f"target_actor_arm{i}.pth")
    torch.save(target_critic1.state_dict(), f"target_critic1_arm{i}.pth")
    torch.save(target_critic2.state_dict(), f"target_critic2_arm{i}.pth")

print("\n所有臂训练结束！")

# 画出 Critic 和 Actor 的损失曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(all_critic_losses, label="Critic Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Critic Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(all_actor_losses, label="Actor Loss", color='orange')
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Actor Loss Curve")
plt.legend()

plt.tight_layout()
plt.show()

