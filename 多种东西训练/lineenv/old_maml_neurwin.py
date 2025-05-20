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

Temperature = 1.0

class Actor(nn.Module):
    def __init__(self, state_dim):
        super(Actor, self).__init__()
        self.input_dim = state_dim + 3
        hidden_dim = 256
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)  
#        self.activation = nn.GELU()
        self.activation=nn.ReLU()
        
        """self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0.00001)
            nn.init.constant_(m.bias, 0.00001)"""
    
    def forward(self, state, env_embed_3d):
        x = torch.cat([state, env_embed_3d], dim=-1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
#        x = self.fc3(x)
#        x = self.activation(x)

        logit = self.fc4(x)
#        logit=torch.clamp(logit,min=-2,max=2)
        return logit

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, env_embed):
        self.buffer.append((state, action, reward, next_state, env_embed))

    def sample_all(self):
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

def compute_mc_actor_loss(actor, transitions, gamma, device):
    states, actions, rewards, _, env_embeds = transitions
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    env_embeds_t = torch.tensor(env_embeds, dtype=torch.float32, device=device)
    batch_size = len(rewards)
    half_batch = batch_size // 2
    returns = np.zeros_like(rewards)
    G = 0.0
    for i in reversed(range(len(rewards))):
        G = rewards[i] + gamma * G
        returns[i] = G
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    logit = actor(states_t, env_embeds_t[:, :3])  # (batch,1)
    lam_val = env_embeds_t[:, 3:4]                # (batch,1)
    # 这里做 (logit - lam) / Temperature
    logit_adj = (logit - lam_val) / Temperature   # (batch,1)
    p1 = torch.sigmoid(logit_adj)                 # (batch,1)

    probs = torch.cat([1 - p1, p1], dim=-1)       # (batch,2)
    log_probs = torch.log(probs + 1e-8)           # (batch,2)

    chosen_log_probs = log_probs.gather(1, actions_t.view(-1,1)).squeeze(-1)  # (batch,)

    actor_loss = - torch.mean(returns_t[:half_batch] * chosen_log_probs[:half_batch])
    return actor_loss

def clone_model(model: nn.Module):
    import copy
    return copy.deepcopy(model)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    N = 100
    OptX = 99
    nb_arms = 50
    prob_values = np.linspace(start=0.2, stop=0.8, num=nb_arms)
    #pq_tasks = [(float(p), float(q)) for p, q in zip(prob_values, reversed(prob_values))]
    pq_tasks=[(float(p), float(p)) for p in prob_values]
    lambda_min = 0
    lambda_max = 2.0
    state_dim = 1

    meta_iterations = 100
    meta_batch_size = 32
    inner_lr = 1e-5       # 内环学习率
    meta_lr = 1e-5       # 外环学习率
    gamma = 0.99

    adaptation_steps_per_task = 256
    meta_steps_per_task = 256
    meta_actor = Actor(state_dim).to(device)
    meta_actor_optim = optim.SGD(meta_actor.parameters(), lr=meta_lr)
    meta_losses_log = []
    for outer_iter in tqdm(range(meta_iterations), desc="Meta Iteration"):
        actor_loss_list = []
        tasks_batch = random.sample(pq_tasks, meta_batch_size)
        meta_buffer = ReplayBuffer()
        adapt_buffer = ReplayBuffer()
        for (p_val, q_val) in tasks_batch:
            lam = random.uniform(lambda_min, lambda_max)
            

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
            actor_fast = clone_model(meta_actor)
            fast_actor_optim = optim.SGD(actor_fast.parameters(), lr=inner_lr)
            #adapt_buffer = ReplayBuffer()
            state = env.reset()
            hhh=random.randint(0,100)
            state=np.array([hhh], dtype=np.intc)
            for _ in range(adaptation_steps_per_task):
                state_arr = np.array(state, dtype=np.float32)
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
            #meta_buffer = ReplayBuffer()
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
    os.makedirs("old_maml_neurwin", exist_ok=True)
    plt.savefig("old_maml_neurwin/loss_curve.png")
    plt.show()

    # 保存最终模型参数
    torch.save(meta_actor.state_dict(), "old_maml_neurwin/meta_actor.pth")

    print("Done. The final meta-params are stored in meta_actor.")
    print("Plots and models are saved in 'maml_sac' folder.")


if __name__ == "__main__":
    main()