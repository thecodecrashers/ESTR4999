import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os

# 进度条
from tqdm import tqdm

# ===========================================
# 1) 导入自定义环境 lineEnv
#    假设你的文件名是 lineEnv.py，里面定义了 class lineEnv
# ===========================================
from lineEnv import lineEnv

# 用于 Actor 输出的温度
Temperature = 1.0

# ===============================
# 1) 只保留 Actor 网络
# ===============================
class Actor(nn.Module):
    """
    Actor 输入: state + (p, q, OptX), 不包含 lambda。
    输出：单个 logit (未经过 sigmoid)，用于动作1的概率。
    """
    def __init__(self, state_dim):
        super(Actor, self).__init__()
        self.input_dim = state_dim + 3

        hidden_dim = 256
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)  # single logit

        self.activation = nn.GELU()

    def forward(self, state, env_embed_3d):
        """
        state:        (batch_size, state_dim)
        env_embed_3d: (batch_size, 3) => (p,q,OptX)
        返回: (batch_size, 1) 的logit
        """
        x = torch.cat([state, env_embed_3d], dim=-1)
        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        logit = self.fc4(x)
#        logit=torch.clamp(logit,min=-2,max=2)
        return logit


# ===============================
# 2) 简单的 ReplayBuffer
# ===============================
class ReplayBuffer:
    """
    这里用于存储 (state, action, reward, next_state, env_embed)，
    但在纯蒙特卡洛里，其实不用 Critic，所以 next_state 用不上了。
    依旧保留是为了尽量不破坏你的原结构。
    """
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, env_embed):
        self.buffer.append((state, action, reward, next_state, env_embed))

    def sample_all(self):
        """
        这里返回全部数据，用于一次性计算蒙特卡洛回报。
        你也可以自己改成只返回固定 batch_size 的数据。
        """
        batch = list(self.buffer)
        states, actions, rewards, next_states, env_embeds = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(env_embeds, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


# ===============================
# 3) 蒙特卡洛方式计算 Actor 的 loss
# ===============================
def compute_mc_actor_loss(actor, transitions, gamma, device):
    """
    给定采集到的多步 (s,a,r,...)，用蒙特卡洛方式（回溯折扣累加）计算总回报，
    并对 Actor 做 REINFORCE-like 更新： L = - E[ G_t * log π(a_t|s_t) ].

    transitions: (states, actions, rewards, next_states, env_embeds)
    """
    states, actions, rewards, _, env_embeds = transitions

    # 转成 torch
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    env_embeds_t = torch.tensor(env_embeds, dtype=torch.float32, device=device)

    # ---- 计算蒙特卡洛回报 G_t （从后往前折扣累加）----
    returns = np.zeros_like(rewards)
    G = 0.0
    # 这里假设没有显式的done标记；若碰到done可以把G清零。
    for i in reversed(range(len(rewards))):
        G = rewards[i] + gamma * G
        returns[i] = G
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    # ---- 计算 log_prob(a_t) ----
    # Actor 只用 env_embed 的前3维 (p,q,OptX)
    logit = actor(states_t, env_embeds_t[:, :3])  # (batch,1)
    lam_val = env_embeds_t[:, 3:4]                # (batch,1)
    # 这里做 (logit - lam) / Temperature
    logit_adj = (logit - lam_val) / Temperature   # (batch,1)
    p1 = torch.sigmoid(logit_adj)                 # (batch,1)

    # 拼成对 [p(a=0), p(a=1)]
    probs = torch.cat([1 - p1, p1], dim=-1)       # (batch,2)
    log_probs = torch.log(probs + 1e-8)           # (batch,2)

    # 取对应action的 log_prob
    chosen_log_probs = log_probs.gather(1, actions_t.view(-1,1)).squeeze(-1)  # (batch,)

    # ---- REINFORCE 的 loss = - E[ G_t * log_prob(a_t) ] ----
    # 这里对整条轨迹的每一步都累加
    actor_loss = - torch.mean(returns_t * chosen_log_probs)
    return actor_loss


# ===============================
# 4) 复制模型 (用于MAML内环)
# ===============================
def clone_model(model: nn.Module):
    import copy
    return copy.deepcopy(model)


# ================================
# 5) 主要 MAML + 蒙特卡洛PG 训练循环
# ================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- 参数 ----------
    N = 100
    OptX = 99

    # 定义一组任务: (p,q) 组合
    nb_arms = 50
    prob_values = np.linspace(start=0.1, stop=0.9, num=nb_arms)
    pq_tasks = [(float(p), float(q)) for p, q in zip(prob_values, reversed(prob_values))]

    # lambda 的最小值和最大值
    lambda_min = 0
    lambda_max = 2.0

    # state_dim = 1 (例如环境中只有一个位置索引当作state)
    state_dim = 1

    # MAML 超参数
    meta_iterations = 1250
    meta_batch_size = 32
    inner_lr = 1e-5       # 内环学习率
    meta_lr = 1e-5         # 外环学习率
    gamma = 0.99

    # 每个任务采集多少步做adaptation & meta
    adaptation_steps_per_task = 256
    meta_steps_per_task = 256

    # 构建 Meta-Actor (只要一个模型)
    meta_actor = Actor(state_dim).to(device)

    # 外环(元参数)优化器
    meta_actor_optim = optim.Adam(meta_actor.parameters(), lr=meta_lr)

    meta_losses_log = []

    # ============= MAML 外环 ============
    for outer_iter in tqdm(range(meta_iterations), desc="Meta Iteration"):
        # 用 list 存储本轮外环所有子任务的 meta-loss
        actor_loss_list = []

        # 1) 从任务集中采样 meta_batch_size 个 (p, q)
        tasks_batch = random.sample(pq_tasks, meta_batch_size)

        for (p_val, q_val) in tasks_batch:
            # 随机抽取一个 lambda
            lam = random.uniform(lambda_min, lambda_max)

            # 构建该任务对应的环境
            env = lineEnv(seed=42, N=N, OptX=OptX, p=p_val, q=q_val)

            # env_embed_4d = (p, q, OptX, lam)
            env_embed_4d = torch.tensor([p_val, q_val, float(OptX), lam],
                                        dtype=torch.float32, device=device)

            # 2) 克隆 meta-params => fast params
            actor_fast = clone_model(meta_actor)
            fast_actor_optim = optim.SGD(actor_fast.parameters(), lr=inner_lr)

            # 3) 收集少量数据(rollout) for adaptation
            adapt_buffer = ReplayBuffer()
            state = env.reset()
            for _ in range(adaptation_steps_per_task):
                state_arr = np.array(state, dtype=np.float32)

                # 策略采样动作
                s_t = torch.from_numpy(state_arr).unsqueeze(0).to(device)  # (1,1)
                with torch.no_grad():
                    logit = actor_fast(s_t, env_embed_4d[:3].unsqueeze(0))  # (1,1)
                    lam_val = env_embed_4d[3].item()
                    p1 = torch.sigmoid((logit - lam_val)/Temperature)       # (1,1)
                    probs = torch.cat([1 - p1, p1], dim=-1)                 # (1,2)
                    dist = torch.distributions.Categorical(probs=probs)
                    action = dist.sample().item()

                # 与环境交互
                next_state, reward, done, _ = env.step(action)
                # 如果动作为1，就减去 lam
                if action == 1:
                    reward -= lam

                adapt_buffer.push(state_arr, action, reward, next_state, 
                                  env_embed_4d.cpu().numpy())

                state = next_state
                if done:
                    state = env.reset()

            # 若收集数据太少，可以跳过
            if len(adapt_buffer) < 10:
                continue

            # 4) 对 fast网络做一次梯度更新 (单步adaptation)
            adapt_data = adapt_buffer.sample_all()  # 取全部
            a_actor_loss = compute_mc_actor_loss(actor_fast, adapt_data, gamma, device)
            fast_actor_optim.zero_grad()
            a_actor_loss.backward()
            fast_actor_optim.step()

            # 5) 用更新后的 fast网络收集 meta 数据
            meta_buffer = ReplayBuffer()
            state = env.reset()
            for _ in range(meta_steps_per_task):
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
                    reward -= lam

                meta_buffer.push(state_arr, action, reward, next_state,
                                 env_embed_4d.cpu().numpy())

                state = next_state
                if done:
                    state = env.reset()

            if len(meta_buffer) < 10:
                continue

            # 6) 计算对 meta-params 的损失 (meta-loss)
            meta_data = meta_buffer.sample_all()
            m_actor_loss = compute_mc_actor_loss(actor_fast, meta_data, gamma, device)
            # 不要在 fast_actor_optim 上 step，而是把梯度回传到 meta_actor
            # (因为 actor_fast 是从 meta_actor clone 出来的，会共享初始参数)
            actor_loss_list.append(m_actor_loss)

        # 7) 外环(meta)参数更新: 将多个任务的损失做平均
        if len(actor_loss_list) > 0:
            meta_actor_loss_val = torch.mean(torch.stack(actor_loss_list))
            meta_actor_optim.zero_grad()
            meta_actor_loss_val.backward()
            meta_actor_optim.step()

            meta_losses_log.append(meta_actor_loss_val.item())
            print(f"[Meta Iter={outer_iter}] loss={meta_actor_loss_val.item():.3f}")
        else:
            meta_losses_log.append(0.0)

    # ============= 画一下 meta-loss 曲线 ============ 
    plt.figure(figsize=(7,5))
    plt.plot(meta_losses_log, label="Meta-Actor Loss (MC-PG)")
    plt.xlabel("Meta-Iteration")
    plt.ylabel("Loss")
    plt.title("MAML + Monte Carlo Policy Gradient (No Critic, No Entropy)")
    plt.legend()

    # 创建保存目录
    os.makedirs("maml_neurwin", exist_ok=True)
    plt.savefig("maml_neurwin/loss_curve.png")
    plt.show()

    # 保存最终模型参数
    torch.save(meta_actor.state_dict(), "maml_neurwin/meta_actor.pth")

    print("Done. The final meta-params are stored in meta_actor.")
    print("Plots and models are saved in 'maml_sac' folder.")


if __name__ == "__main__":
    main()
