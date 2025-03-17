import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from lineEnv import lineEnv

Temperature=1.0

class Actor(nn.Module):
    def __init__(self, state_dim):
        super(Actor, self).__init__()
        self.input_dim = state_dim + 3
        hidden_dim = 256  

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)  # single logit

        self.activation = nn.GELU()

        # 应用自定义初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, -0.01)  # 权重初始化为0
            nn.init.constant_(m.bias, -0.01)    # 偏置初始化为0

    def forward(self, state, env_embed_3d):
        x = torch.cat([state, env_embed_3d], dim=-1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        logit = self.fc5(x)  
#        logit = torch.clamp(logit, min=-2, max=2)
        return logit



class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.input_dim = state_dim + 4
        hidden_dim = 256  

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 2)  # Q(s, a=0) 和 Q(s, a=1)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        # 应用自定义初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, -0.01)  # 权重初始化为0
            nn.init.constant_(m.bias, -0.01)    # 偏置初始化为0

    def forward(self, state, env_embed_4d):
        x = torch.cat([state, env_embed_4d], dim=-1)  

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.activation(x)
        q_vals = self.fc5(x)  
        return q_vals



class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, env_embed):
        # env_embed 这里依旧可以是 4 维 (p,q,OptX,lambda)
        self.buffer.append((state, action, reward, next_state, env_embed))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, env_embeds = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(env_embeds))

    def __len__(self):
        return len(self.buffer)

def compute_sac_losses(actor, critic, batch, gamma, alpha, device):
    states, actions, rewards, next_states, env_embeds = batch

    # 转成tensor
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    env_embeds_t = torch.tensor(env_embeds, dtype=torch.float32, device=device)

    # ---------- Critic 计算 Q(s,a) ----------
    q_values = critic(states_t, env_embeds_t)  # (batch,2)
    q_values=torch.clamp(q_values,min=-200,max=200)
    q_action = q_values.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)  # (batch,)

    # ---------- 目标 Q: r + gamma * V(s') ----------
    with torch.no_grad():
        next_q_values = critic(next_states_t, env_embeds_t)         # (batch,2)
        next_q_values=torch.clamp(next_q_values,min=-200,max=200)
        # Actor 用 env_embed_3d = env_embed_4d[:, :3]
        next_logit = actor(next_states_t, env_embeds_t[:, :3])      # (batch,1)
        lam_next = env_embeds_t[:, 3:4]                             # (batch,1)
        # 注意这里做 logit - lambda
        next_p1 = torch.sigmoid((next_logit - lam_next)/Temperature)             # (batch,1)
        next_probs = torch.cat([1 - next_p1, next_p1], dim=-1)      # (batch,2)

        next_log_probs = torch.log(next_probs + 1e-8)

        # soft state value
        next_q = torch.sum(next_probs * next_q_values, dim=-1)         # (batch,)
        next_entropy = -torch.sum(next_probs * next_log_probs, dim=-1) # (batch,)
        target_v = next_q + alpha * next_entropy
    target_q = rewards_t + gamma * target_v
    critic_loss = nn.MSELoss()(q_action, target_q)

    # ---------- Actor部分 ----------
    # Actor 输入: (state, env_embed_4d[:, :3])
    logit = actor(states_t, env_embeds_t[:, :3])     # (batch,1)
    lam_val = env_embeds_t[:, 3:4]                   # (batch,1)
    # 这里做 logit - lambda
    p1 = torch.sigmoid((logit - lam_val)/Temperature)              # (batch,1)
    probs = torch.cat([1 - p1, p1], dim=-1)          # (batch,2)
    log_probs = torch.log(probs+1e-8)

    q_vals = critic(states_t, env_embeds_t)          # (batch,2)
    q_expected = torch.sum(probs * q_vals, dim=-1)   # (batch,)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (batch,)
    actor_objective = q_expected + alpha * entropy
    actor_loss = -torch.mean(actor_objective)
    return actor_loss, critic_loss


def clone_model(model: nn.Module):

    import copy
    return copy.deepcopy(model)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- 参数 ----------
    N = 100
    OptX = 99

    # 定义一组任务: (p,q) 组合
    nb_arms = 50
    prob_values = np.linspace(start=0.1, stop=0.9, num=nb_arms)
    #pq_tasks = [(float(p), float(q)) for p, q in zip(prob_values, reversed(prob_values))]
    pq_tasks=[(float(p), float(p)) for p in prob_values]
    # 新增：lambda的最小值和最大值
    lambda_min = 0
    lambda_max = 2.0

    # 现在 state_dim = 1 (比如线性环境里，可能只有一个位置/索引的 state)
    state_dim = 1

    # MAML超参数
    meta_iterations = 10000
    meta_batch_size = 32
    inner_lr = 0.0001        # 内环学习率
    meta_lr = 0.0001              # 外环学习率
    gamma = 0.99
    alpha = 0.1

    # 每个任务采集多少步做adaptation & meta
    adaptation_steps_per_task = 128
    meta_steps_per_task = 128

    batch_size = 128
    memory_size = 500000  # replay buffer容量

    # 构建Meta-Actor & Meta-Critic
    # Actor 不输入 lambda, Critic 输入 lambda
    meta_actor = Actor(state_dim).to(device)
    meta_critic = Critic(state_dim).to(device)

    # 外环(元参数)优化器
    meta_actor_optim = optim.Adam(meta_actor.parameters(), lr=meta_lr)
    meta_critic_optim = optim.Adam(meta_critic.parameters(), lr=meta_lr)

    meta_losses_log = []

    # ============= MAML外环 ============
    for outer_iter in tqdm(range(meta_iterations), desc="Meta Iteration"):
        # 用 list 存储本轮外环所有子任务的 meta-loss
        actor_loss_list = []
        critic_loss_list = []

        # 1) 从任务集中采样 meta_batch_size 个 (p, q)
        tasks_batch = random.sample(pq_tasks, meta_batch_size)

        for (p_val, q_val) in tasks_batch:
            # 随机抽取一个lambda
            lam = random.uniform(lambda_min, lambda_max)
            
            # 额外添加：随机选择一个状态
            ss = np.random.randint(1, 101)  # 这里假设状态是 [1, 100] 的整数
            ss=np.array([ss], dtype=np.intc)
            ss = np.array(ss, dtype=np.float32)
            ss = torch.from_numpy(ss).to(device).unsqueeze(0)  # 转换为 tensor
            with torch.no_grad():
                value = meta_actor(ss, torch.tensor([p_val, q_val, float(OptX)], dtype=torch.float32, device=device).unsqueeze(0))
            # 确保 value 是一个 Python 数值
            if isinstance(value, torch.Tensor):
                value = value.item()
            # 如果 value 在 lambda 允许的范围内，则替换 lam
            if lambda_min >value or value> lambda_max:
                lam = value
           
            env = lineEnv(seed=42, N=N, OptX=OptX, p=p_val, q=q_val)
            env_embed_4d = torch.tensor([p_val, q_val, float(OptX), lam],
                                        dtype=torch.float32, device=device)
            # 2) 克隆 meta-params => fast params
            actor_fast = clone_model(meta_actor)
            critic_fast = clone_model(meta_critic)
            fast_actor_optim = optim.Adam(actor_fast.parameters(), lr=inner_lr)
            fast_critic_optim = optim.Adam(critic_fast.parameters(), lr=inner_lr)

            # 3) 收集少量数据(rollout) for adaptation
            adapt_buffer = ReplayBuffer(max_size=memory_size)
            state = env.reset()
            for hh in range(adaptation_steps_per_task):
                state_arr = np.array(state, dtype=np.float32)
                s_t = torch.from_numpy(state_arr).unsqueeze(0).to(device)  # (1,1)

                with torch.no_grad():
                    # Actor forward 不包含 lam
                    logit = actor_fast(s_t, env_embed_4d[:3].unsqueeze(0))  # (1,1)
                    # 在这一步减去 lam
                    lam_val = env_embed_4d[3].item()
                    p1 = torch.sigmoid((logit - lam_val)/Temperature)                     # (1,1)
                    probs = torch.cat([1 - p1, p1], dim=-1)  # (1,2)
                    dist = torch.distributions.Categorical(probs=probs)
                    action = dist.sample().item()

                # 与环境交互
                next_state, reward, done, _ = env.step(action)
                # 如果动作为1，就减去 lambda
                if action == 1:
                    reward = reward - lam

                adapt_buffer.push(
                    state_arr, 
                    action, 
                    reward, 
                    next_state, 
                    env_embed_4d.cpu().numpy()  # 存4维
                )
                state = hh%100
                state=np.array([state],dtype=np.intc)
                

            # 如果收集的数据太少，跳过
            if len(adapt_buffer) < batch_size:
                continue

            # 4) 对 fast网络做一次梯度更新 (单步adaptation)
            batch_adapt = adapt_buffer.sample(batch_size)
            a_actor_loss, a_critic_loss = compute_sac_losses(
                actor_fast, critic_fast, batch_adapt, gamma, alpha, device
            )
            fast_actor_optim.zero_grad()
            fast_critic_optim.zero_grad()
            a_actor_loss.backward(retain_graph=True)
            a_critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(actor_fast.parameters(), max_norm=10.0)
            torch.nn.utils.clip_grad_norm_(critic_fast.parameters(), max_norm=10.0)
            fast_actor_optim.step()
            fast_critic_optim.step()

            # 5) 用更新后的 fast网络收集 meta数据
            meta_buffer = ReplayBuffer(max_size=memory_size)
            state = env.reset()
            for hh in range(meta_steps_per_task):
                state_arr = np.array(state, dtype=np.float32)
                s_t = torch.from_numpy(state_arr).unsqueeze(0).to(device)

                with torch.no_grad():
                    logit = actor_fast(s_t, env_embed_4d[:3].unsqueeze(0))
                    lam_val = env_embed_4d[3].item()
                    p1 = torch.sigmoid((logit - lam_val)/Temperature)
                    probs = torch.cat([1 - p1, p1], dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    action = dist.sample().item()

                next_state, reward, done, _ = env.step(action)
                if action == 1:
                    reward = reward - lam

                meta_buffer.push(
                    state_arr,
                    action,
                    reward,
                    next_state,
                    env_embed_4d.cpu().numpy()
                )
                state = hh%100
                state=np.array([state],dtype=np.intc)
                if done:
                    state = env.reset()

            # 6) 在 fast网络上计算 meta-loss，但梯度回传到 meta-params
            batch_meta = meta_buffer.sample(batch_size)
            m_actor_loss, m_critic_loss = compute_sac_losses(
                actor_fast, critic_fast, batch_meta, gamma, alpha, device
            )
            # 把子任务的 loss 直接保存到 list (保留在计算图中)
            actor_loss_list.append(m_actor_loss)
            critic_loss_list.append(m_critic_loss)

        # 7) 外环(meta)参数更新: 将多个任务的损失做平均
        if len(actor_loss_list) > 0:
            meta_actor_loss_val = torch.mean(torch.stack(actor_loss_list))
            meta_critic_loss_val = torch.mean(torch.stack(critic_loss_list))
            meta_loss = meta_actor_loss_val + meta_critic_loss_val
            meta_actor_optim.zero_grad()
            meta_critic_optim.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_actor.parameters(), max_norm=10.0)
            torch.nn.utils.clip_grad_norm_(meta_critic.parameters(), max_norm=10.0)
            meta_actor_optim.step()
            meta_critic_optim.step()
            meta_losses_log.append((meta_actor_loss_val.item(), meta_critic_loss_val.item()))
            print(meta_actor_loss_val.item())
            print(meta_critic_loss_val.item())
        else:
            # 如果没有任何有效 batch，则记个0
            meta_losses_log.append((0.0, 0.0))

    # ============= 画一下meta-loss曲线 ============ 
    actor_losses, critic_losses = zip(*meta_losses_log)
    plt.figure(figsize=(7,5))
    plt.plot(actor_losses, label="Meta-Actor Loss")
    plt.plot(critic_losses, label="Meta-Critic Loss")
    plt.xlabel("Meta-Iteration")
    plt.ylabel("Loss")
    plt.title("MAML (Discrete SAC) - Actor(logit - lambda)")
    plt.legend()

    # 创建保存目录
    os.makedirs("maml_sac", exist_ok=True)
    # 保存图像
    plt.savefig("maml_sac/loss_curve.png")
    plt.show()

    # 保存最终的模型参数
    torch.save(meta_actor.state_dict(), "maml_sac/meta_actor.pth")
    torch.save(meta_critic.state_dict(), "maml_sac/meta_critic.pth")

    print("Done. The final meta-params are stored in meta_actor, meta_critic.")
    print("Plots and models are saved in 'maml_sac' folder.")

if __name__ == "__main__":
    main()
