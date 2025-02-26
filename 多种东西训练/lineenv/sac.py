import torch
import torch.nn as nn
import numpy as np
import random
import lineEnv  # 你的自定义环境
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
import os

# 自动检测 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# 超参数
# ------------------------------
state_dim = 1
env_features_dim = 3
embed_dim = 16
nb_arms = 15

gamma = 0.99
batch_size = 16
memory_size = 1000

num_episodes = 10
max_steps_per_episode = 20

# SAC 常用
alpha = 0.2          # 熵系数(固定)
learning_rate = 1e-3
tau = 0.005          # 软更新系数

# 生成 (p, q) 组合
prob_values = np.linspace(start=0.1, stop=0.9, num=nb_arms)
pq_selected = [(p, q) for p, q in zip(prob_values, reversed(prob_values))]

# ------------------------------
# 经验回放缓冲区
# ------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.tensor(action, dtype=torch.long, device=device).unsqueeze(-1),
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
    """
    处理环境额外特征 (3维 => 16维)，再与 state 拼接。
    """
    def __init__(self, input_dim, output_dim):
        super(EmbeddingNet, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class Actor(nn.Module):
    """
    离散动作 (0/1) 的 SAC:
    - 这里让 Actor 输出 1 个 logit => p = sigmoid(logit)
      => π(0|s)= 1-p, π(1|s)= p
    """
    def __init__(self, state_dim, env_embed_dim):
        super(Actor, self).__init__()
        self.input_dim = state_dim + env_embed_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # 输出一个 logit

        self.relu = nn.ReLU()

    def forward(self, state, env_embed):
        x = torch.cat([state, env_embed], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logit = self.fc3(x)  # (batch,1)
        return logit  # 后续可做 sigmoid -> p

    def get_action_probs(self, state, env_embed):
        logit = self.forward(state, env_embed)  # (batch,1)
        p = torch.sigmoid(logit)                # (batch,1)
        probs = torch.cat([1 - p, p], dim=-1)   # (batch,2)
        return probs


class Critic(nn.Module):
    """
    输出 Q(s, a=0) 和 Q(s, a=1)，shape=(batch,2)
    """
    def __init__(self, state_dim, env_embed_dim):
        super(Critic, self).__init__()
        self.input_dim = state_dim + env_embed_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # 输出2个动作的Q值

        self.relu = nn.ReLU()

    def forward(self, state, env_embed):
        """
        返回 Q(s)=[Q(s,0), Q(s,1)] of shape (batch,2)
        """
        x = torch.cat([state, env_embed], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_vals = self.fc3(x)  # (batch,2)
        return q_vals

def soft_update(source_net, target_net, tau):
    for tp, sp in zip(target_net.parameters(), source_net.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

# ------------------------------
# 预训练嵌入层 (共享)
# ------------------------------
pretrain_epochs = 5  # 训练轮数
pretrain_episodes = 50  # 每个臂的采样 episode 数
pretrain_batch_size = 32  # 预训练的 batch 大小
memory_size = 1000  # 经验回放容量

print("\n========== 预训练嵌入层 (共享) ==========")

# **创建共享的嵌入层**
embedding_net = EmbeddingNet(env_features_dim, embed_dim).to(device)
embedding_optimizer = torch.optim.Adam(embedding_net.parameters(), lr=learning_rate)

# **为每个臂创建独立的 Critic 和 ReplayBuffer**
arms_data = []
for i, (p, q) in enumerate(pq_selected):
    env = lineEnv.lineEnv(seed=42, N=10, OptX=99, p=p, q=q)

    # 该臂独立的 Critic
    critic1 = Critic(state_dim, embed_dim).to(device)
    critic2 = Critic(state_dim, embed_dim).to(device)
    target_critic1 = Critic(state_dim, embed_dim).to(device)
    target_critic2 = Critic(state_dim, embed_dim).to(device)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    # 该臂独立的优化器
    critic1_optimizer = torch.optim.Adam(critic1.parameters(), lr=learning_rate)
    critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=learning_rate)

    # 该臂独立的经验回放
    memory = ReplayBuffer(memory_size)

    # **采集经验**
    for _ in range(pretrain_episodes):
        state_val = env.reset()
        done = False
        step_count = 0

        while not done and step_count < max_steps_per_episode:
            action_int = env.action_space.sample()  # 随机动作
            next_state_val, reward, done, _ = env.step(action_int)
            memory.push(state_val, action_int, reward, next_state_val, done)
            state_val = next_state_val
            step_count += 1

    arms_data.append({
        "env": env,
        "critic1": critic1,
        "critic2": critic2,
        "target_critic1": target_critic1,
        "target_critic2": target_critic2,
        "critic1_optimizer": critic1_optimizer,
        "critic2_optimizer": critic2_optimizer,
        "memory": memory,
        "p": p,
        "q": q
    })

print(f"所有臂的经验收集完成，开始训练嵌入层！")

# 训练循环
embedding_losses = []

pbar = tqdm(range(pretrain_epochs), desc=f"Pretraining Shared Embedding")

for epoch in pbar:
    for arm in arms_data:
        memory = arm["memory"]
        critic1 = arm["critic1"]
        critic2 = arm["critic2"]
        target_critic1 = arm["target_critic1"]
        target_critic2 = arm["target_critic2"]
        critic1_optimizer = arm["critic1_optimizer"]
        critic2_optimizer = arm["critic2_optimizer"]
        p, q = arm["p"], arm["q"]

        if len(memory) < pretrain_batch_size:
            continue  # 跳过数据不足的臂

        # **训练批次**
        for _ in range(len(memory) // pretrain_batch_size):
            s_b, a_b, r_b, s2_b, d_b = memory.sample(pretrain_batch_size)
            s_b=s_b.squeeze(-1)
            s2_b=s2_b.squeeze(-1)
            # **计算共享的环境嵌入**
            env_features = torch.tensor([99, p, q], dtype=torch.float32, device=device).unsqueeze(0)
            env_embed_full = embedding_net(env_features).detach()
            env_embed_b = env_embed_full.repeat(pretrain_batch_size, 1)

            # **计算 Critic 目标值**
            with torch.no_grad():
                q1_next = target_critic1(s2_b, env_embed_b)
                q2_next = target_critic2(s2_b, env_embed_b)
                q_min_next = torch.min(q1_next, q2_next)
                y = r_b + gamma * (1 - d_b) * q_min_next.gather(1, a_b)

            # **计算 Critic 损失**
            q1_vals = critic1(s_b, env_embed_b)
            q2_vals = critic2(s_b, env_embed_b)
            q1_chosen = q1_vals.gather(1, a_b)
            q2_chosen = q2_vals.gather(1, a_b)

            critic1_loss = nn.MSELoss()(q1_chosen, y)
            critic2_loss = nn.MSELoss()(q2_chosen, y)
            embedding_loss = (critic1_loss + critic2_loss) * 0.5

            # **只优化 `EmbeddingNet`，不优化 Critic**
            embedding_optimizer.zero_grad()
            embedding_loss.backward()
            embedding_optimizer.step()

            embedding_losses.append(embedding_loss.item())

            # **软更新 Critic**
            soft_update(critic1, target_critic1, tau)
            soft_update(critic2, target_critic2, tau)

    pbar.set_postfix({
        "EmbeddingLoss": f"{embedding_losses[-1]:.4f}" if embedding_losses else "N/A"
    })

# 训练结束后，保存共享的嵌入层
torch.save(embedding_net.state_dict(), os.path.join("sac", "shared_pretrained_embedding.pth"))

print(f"预训练完成，嵌入层已保存为 shared_pretrained_embedding.pth")



# ------------------------------
# 主训练循环 (离散动作 SAC)
# ------------------------------
all_critic_losses = []
all_actor_losses = []

for i, (p, q) in enumerate(pq_selected):
    print(f"\n========== 训练臂 {i}: p={p:.2f}, q={q:.2f} ==========")

    # 创建环境
    env = lineEnv.lineEnv(seed=42, N=10, OptX=99, p=p, q=q)

    # 嵌入网络
#    embedding_net = EmbeddingNet(env_features_dim, embed_dim).to(device)

    # 两个Critic(用于最小裁剪) + Actor
    critic1 = Critic(state_dim, embed_dim).to(device)
    critic2 = Critic(state_dim, embed_dim).to(device)
    actor = Actor(state_dim, embed_dim).to(device)

    # 目标网络
    target_critic1 = Critic(state_dim, embed_dim).to(device)
    target_critic2 = Critic(state_dim, embed_dim).to(device)

    # 同步初始参数
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    # 优化器
    critic1_optimizer = torch.optim.Adam(critic1.parameters(), lr=learning_rate)
    critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=learning_rate)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)

    # 经验回放
    memory = ReplayBuffer(memory_size)

    critic_losses = []
    actor_losses_ = []

    # 预先准备环境特征的 embedding（这里假设固定：99, p, q）
    env_features = torch.tensor([99, p, q], dtype=torch.float32, device=device).unsqueeze(0)
    # shape=(1, 3)
    env_embed_full = embedding_net(env_features).detach()  # shape=(1, embed_dim)

    # -------------------- 训练多个 Episode --------------------
    pbar = tqdm(range(num_episodes), desc=f"Training Arm {i}")
    for episode in pbar:
        state_val = env.reset()
        state = torch.tensor([state_val], dtype=torch.float32, device=device) # (1,1)

        episode_reward = 0.0
        for step in range(max_steps_per_episode):
            # 1) 与环境交互：按照当前策略概率分布选动作
            with torch.no_grad():
                # 计算当前状态下动作=1的概率 p
                logit = actor(state, env_embed_full)  # shape=(1,1)
                p_1 = torch.sigmoid(logit)            # shape=(1,1)
                # Bernoulli采样得到离散动作
                action_sample = torch.bernoulli(p_1).long()  # (1,1), 值=0或1
                action_int = int(action_sample.item())

            next_state_val, reward, done, _ = env.step(action_int)

            # 存储
            memory.push(state_val, action_int, reward, next_state_val, done)

            state_val = next_state_val
            state = torch.tensor([state_val], dtype=torch.float32, device=device)
            episode_reward += reward

            # 2) SAC 更新
            if len(memory) >= batch_size:
                # ----- 2.1 从回放中取一个batch -----
                s_b, a_b, r_b, s2_b, d_b = memory.sample(batch_size)
                # env_embed 批量复制
                s2_b=s2_b.squeeze(-1)
                env_embed_b = env_embed_full.repeat(batch_size, 1)
                # ----- 2.2 计算Critic的目标Q值 y -----
                with torch.no_grad():
                    # 下一时刻的动作分布
                    probs_next = actor.get_action_probs(s2_b, env_embed_b)  # (batch,2) => pi(a'=0|s'), pi(a'=1|s')
                    # 目标Q
                    q1_next = target_critic1(s2_b, env_embed_b)  # (batch,2)
                    q2_next = target_critic2(s2_b, env_embed_b)  # (batch,2)
                    q_min_next = torch.min(q1_next, q2_next)     # (batch,2)

                    # V(s') = \sum_{a'} pi(a'|s') [ Q_min(s',a') - alpha * log pi(a'|s') ]
                    log_probs_next = torch.log(probs_next + 1e-8)     # (batch,2)
                    inside_term = q_min_next - alpha * log_probs_next # (batch,2)
                    V_next = (probs_next * inside_term).sum(dim=1, keepdim=True)  # (batch,1)

                    # TD 目标
                    y = r_b + gamma * (1 - d_b) * V_next

                # ----- 2.3 更新 Critic1, Critic2 -----
                # Q1, Q2 各输出 (batch,2). 取出对应动作 a_b 的 Q(s,a)
                s_b=s_b.squeeze(-1) # (batch,
                q1_vals = critic1(s_b, env_embed_b)  # (batch,2)
                q2_vals = critic2(s_b, env_embed_b)  # (batch,2)
                # gather the Q for chosen action
                q1_chosen = q1_vals.gather(1, a_b)    # (batch,1)
                q2_chosen = q2_vals.gather(1, a_b)    # (batch,1)

                critic1_loss = nn.MSELoss()(q1_chosen, y)
                critic2_loss = nn.MSELoss()(q2_chosen, y)

                critic1_optimizer.zero_grad()
                critic1_loss.backward()
                critic1_optimizer.step()

                critic2_optimizer.zero_grad()
                critic2_loss.backward()
                critic2_optimizer.step()

                critic_loss = (critic1_loss.item() + critic2_loss.item()) * 0.5
                critic_losses.append(critic_loss)

                # ----- 2.4 更新 Actor -----
                # 离散动作 SAC 中的 actor_loss:
                #   J(\pi) = E_s [ sum_{a} pi(a|s) ( alpha log pi(a|s) - Qmin(s,a) ) ]
                probs_curr = actor.get_action_probs(s_b, env_embed_b)    # (batch,2)
                log_probs_curr = torch.log(probs_curr + 1e-8)           # (batch,2)

                q1_curr = critic1(s_b, env_embed_b)                     # (batch,2)
                q2_curr = critic2(s_b, env_embed_b)                     # (batch,2)
                q_min_curr = torch.min(q1_curr, q2_curr)                # (batch,2)

                inside_term_actor = alpha * log_probs_curr - q_min_curr # (batch,2)
                actor_loss = (probs_curr * inside_term_actor).sum(dim=1).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                actor_losses_.append(actor_loss.item())

                # ----- 2.5 软更新 target critic -----
                soft_update(critic1, target_critic1, tau)
                soft_update(critic2, target_critic2, tau)

            if done:
                break

        pbar.set_postfix({
            "EpisodeReward": f"{episode_reward:.2f}",
            "CriticLoss": f"{critic_losses[-1] if len(critic_losses)>0 else 0:.4f}",
            "ActorLoss": f"{actor_losses_[-1] if len(actor_losses_)>0 else 0:.4f}"
        })

    # 记录该臂训练过程的损失
    all_critic_losses.extend(critic_losses)
    all_actor_losses.extend(actor_losses_)

    # 保存模型
    torch.save(actor.state_dict(), os.path.join("sac",f"actor_arm{i}.pth"))
    torch.save(critic1.state_dict(), os.path.join("sac",f"critic1_arm{i}.pth"))
    torch.save(critic2.state_dict(), os.path.join("sac",f"critic2_arm{i}.pth"))
    torch.save(target_critic1.state_dict(), os.path.join("sac",f"target_critic1_arm{i}.pth"))
    torch.save(target_critic2.state_dict(), os.path.join("sac",f"target_critic2_arm{i}.pth"))

print("\n所有臂训练结束！")

# 画出 Critic 与 Actor 的损失曲线
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
