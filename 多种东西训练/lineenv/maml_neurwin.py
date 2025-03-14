import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Suppose you have lineEnv
from lineEnv import lineEnv

Temperature = 1.0
gamma       = 0.99

class Actor(nn.Module):
    def __init__(self, state_dim):
        super(Actor, self).__init__()
        self.input_dim = state_dim + 3
        hidden_dim = 256  
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1) 
        self.activation = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0.01)
            nn.init.constant_(m.bias, 0.01)

    def forward(self, state, env_embed_3d):
        """
        state: (batch, state_dim)
        env_embed_3d: (batch, 3) => (p, q, OptX)
        """
        x = torch.cat([state, env_embed_3d], dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        logit = self.fc5(x)
        return logit


def clone_model(model: nn.Module):
    import copy
    return copy.deepcopy(model)


def rollout_one_episode(actor, env, start_state, lam, gamma, device, max_steps=200):
    """
    Roll out an episode from a specific start_state, returning:
       sum_{t}[ G_t * log_prob(a_t) ].
    """
    # Reset the environment and force the state to `start_state`.
    env.reset()
    env.state = start_state

    log_probs = []
    rewards   = []

    for step_i in range(max_steps):
        # -- Observe current state as a torch tensor (requires_grad=False is fine).
        s_t = torch.tensor([env.state], dtype=torch.float32, device=device)
        
        # -- Forward pass through actor (NO 'with torch.no_grad()'!)
        logit = actor(
            s_t.unsqueeze(0),
            torch.tensor([env.p, env.q, float(env.OptX)], 
                         dtype=torch.float32, device=device).unsqueeze(0)
        )
        
        # -- Adjust by lambda, then form probabilities
        adj_logit = (logit - lam) / Temperature
        p1 = torch.sigmoid(adj_logit)           
        probs = torch.cat([1 - p1, p1], dim=-1)  # shape (1,2)
        
        # -- Sample action
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()             # shape (), an int {0,1}
        log_prob = dist.log_prob(action)   # shape ()

        # -- Step environment
        next_state, reward, done, _ = env.step(action.item())
        # subtract lam if action=1
        if action.item() == 1:
            reward -= lam

        log_probs.append(log_prob)
        rewards.append(reward)

        if done:
            break

    T = len(rewards)
    if T == 0:
        return torch.tensor(0.0, device=device, requires_grad=False)

    # Convert rewards => torch for discounted returns
    rewards_t  = torch.tensor(rewards, dtype=torch.float32, device=device)
    log_probs_t = torch.stack(log_probs)  # shape (T,)

    # Compute discounted returns
    returns_t = torch.zeros(T, dtype=torch.float32, device=device)
    running_return = 0.0
    for t in reversed(range(T)):
        running_return = rewards_t[t] + gamma * running_return
        returns_t[t] = running_return

    # sum_{t=0..T-1}[ G_t * log_prob(a_t) ]
    total_objective = torch.sum(returns_t * log_probs_t)
    return total_objective

import torch
import concurrent.futures
import os

def reinforce_loss_over_all_states(actor, env, lam, gamma, device, N, max_steps=200):
    """
    Use multithreading to parallelize rollouts for multiple states.
    """
    total_obj = torch.tensor(0.0, device=device)  # Keep as torch scalar

    # Get the number of available CPU cores
    num_threads = min(os.cpu_count(), N)  # Use up to N threads or max available CPU cores
    
    # Define a function wrapper for parallel execution
    def rollout_wrapper(s):
        return rollout_one_episode(actor, env, s, lam, gamma, device, max_steps)

    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(rollout_wrapper, range(N)))

    # Sum up all results
    total_obj = torch.stack(results).sum()  # Ensure gradient flow remains intact

    return total_obj


"""
def reinforce_loss_over_all_states(actor, env, lam, gamma, device, N, max_steps=200):
    total_obj = torch.tensor(0.0, device=device)  # keep it as a torch scalar
    for s in range(N):
        ep_obj = rollout_one_episode(actor, env, s, lam, gamma, device, max_steps)
        total_obj = total_obj + ep_obj
    return total_obj"""
 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Hyperparams
    N = 100
    OptX = 99
    nb_arms = 50
    prob_values = np.linspace(start=0.1, stop=0.9, num=nb_arms)
    pq_tasks = [(float(p), float(p)) for p in prob_values]
    lambda_min = 0.0
    lambda_max = 2.0

    state_dim = 1
    meta_iterations = 200
    meta_batch_size = 4   # a bit smaller for speed
    inner_lr = 0.01
    meta_lr  = 0.01
    gamma = 0.99

    adaptation_steps_per_task = 2   # how many gradient steps to adapt
    meta_steps_per_task       = 1   # after adaptation, how many times we measure meta-objective
    max_rollout_len = 50     # horizon for each rollout

    # Build meta actor
    meta_actor = Actor(state_dim).to(device)
    meta_actor_optim = optim.Adam(meta_actor.parameters(), lr=meta_lr)

    meta_losses_log = []

    for outer_iter in tqdm(range(meta_iterations), desc="Meta Iteration"):
        actor_loss_list = []

        # sample tasks
        tasks_batch = random.sample(pq_tasks, meta_batch_size)

        for (p_val, q_val) in tasks_batch:
            lam = random.uniform(lambda_min, lambda_max)

            # Construct environment for this sub-task
            env = lineEnv(seed=42, N=N, OptX=OptX, p=p_val, q=q_val)

            # 1) Clone meta-actor => fast actor
            actor_fast = clone_model(meta_actor)
            fast_actor_optim = optim.Adam(actor_fast.parameters(), lr=inner_lr)

            # 2) Inner loop: adapt actor_fast
            for _ in range(adaptation_steps_per_task):
                fast_actor_optim.zero_grad()
                # big_objective = sum_{s=0..N-1}[ sum_{t}[G_t logπ(a_t|s_t)] ]
                big_objective = reinforce_loss_over_all_states(
                    actor_fast, env, lam, gamma, device, N, max_steps=max_rollout_len
                )
                adapt_loss = -big_objective  # gradient ascent => negate objective
                adapt_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_fast.parameters(), 10)
                fast_actor_optim.step()

            # 3) Meta-objective: after adaptation, measure performance again
            meta_obj = reinforce_loss_over_all_states(
                actor_fast, env, lam, gamma, device, N, max_steps=max_rollout_len
            )
            # We'll store negative for gradient descent on meta-actor
            actor_loss_list.append(-meta_obj)

        # 4) Combine all meta losses from tasks, do a single update on meta-actor
        if len(actor_loss_list) > 0:
            meta_loss_val = torch.mean(torch.stack(actor_loss_list))
            meta_actor_optim.zero_grad()
            meta_loss_val.backward()
            torch.nn.utils.clip_grad_norm_(meta_actor.parameters(), 10)
            meta_actor_optim.step()

            meta_losses_log.append(meta_loss_val.item())
            print(f"[Meta Iter={outer_iter}] MetaLoss={meta_loss_val.item():.3f}")
        else:
            meta_losses_log.append(0.0)

    # Plot meta-loss
    plt.figure(figsize=(7,5))
    plt.plot(meta_losses_log, label="Meta-Actor Loss (REINFORCE)")
    plt.xlabel("Meta-Iteration")
    plt.ylabel("Loss")
    plt.title("MAML + REINFORCE (No Replay Buffer)")
    plt.legend()
    os.makedirs("maml_neurwin", exist_ok=True)
    plt.savefig("maml_neurwin/loss_curve.png")
    plt.show()

    # Save final model
    torch.save(meta_actor.state_dict(), "maml_neurwin/meta_actor.pth")

if __name__ == "__main__":
    main()




"""import math
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
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1) 
        self.activation = nn.GELU()
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0.01)  # 权重初始化为0
            nn.init.constant_(m.bias, 0.01)    # 偏置初始化为0
    def forward(self, state, env_embed_3d):
        x = torch.cat([state, env_embed_3d], dim=-1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        logit = self.fc5(x)  
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

    # 转成 torch
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    env_embeds_t = torch.tensor(env_embeds, dtype=torch.float32, device=device)

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

    # 拼成对 [p(a=0), p(a=1)]
    probs = torch.cat([1 - p1, p1], dim=-1)       # (batch,2)
    log_probs = torch.log(probs + 1e-8)           # (batch,2)

    # 取对应action的 log_prob
    chosen_log_probs = log_probs.gather(1, actions_t.view(-1,1)).squeeze(-1)  # (batch,)

    # ---- REINFORCE 的 loss = - E[ G_t * log_prob(a_t) ] ----
    # 这里对整条轨迹的每一步都累加
    actor_loss = - torch.mean(returns_t * chosen_log_probs)
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
    prob_values = np.linspace(start=0.1, stop=0.9, num=nb_arms)
    #pq_tasks = [(float(p), float(q)) for p, q in zip(prob_values, reversed(prob_values))]
    pq_tasks=[(float(p), float(p)) for p in prob_values]
    # lambda 的最小值和最大值
    lambda_min = 0
    lambda_max = 2.0
    state_dim = 1
    meta_iterations = 10
    meta_batch_size = 32
    inner_lr = 0.01    # 内环学习率
    meta_lr = 0.01       # 外环学习率
    gamma = 0.99
    adaptation_steps_per_task = 128
    meta_steps_per_task = 128

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
            # 构建该任务对应的环境
            env = lineEnv(seed=42, N=N, OptX=OptX, p=p_val, q=q_val)

            # env_embed_4d = (p, q, OptX, lam)
            env_embed_4d = torch.tensor([p_val, q_val, float(OptX), lam],
                                        dtype=torch.float32, device=device)

            # 2) 克隆 meta-params => fast params
            actor_fast = clone_model(meta_actor)
            fast_actor_optim = optim.Adam(actor_fast.parameters(), lr=inner_lr)

            # 3) 收集少量数据(rollout) for adaptation
            adapt_buffer = ReplayBuffer()
            state = env.reset()
            for hh in range(adaptation_steps_per_task):
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

                state = hh%100
                state=np.array([state],dtype=np.intc)

            adapt_data = adapt_buffer.sample_all() 
            a_actor_loss = compute_mc_actor_loss(actor_fast, adapt_data, gamma, device)
            fast_actor_optim.zero_grad()
            a_actor_loss.backward()
            torch.nn.utils.clip_grad_norm(actor_fast.parameters(), 10)
            fast_actor_optim.step()

            # 5) 用更新后的 fast网络收集 meta 数据
            meta_buffer = ReplayBuffer()
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
                    reward -= lam

                meta_buffer.push(state_arr, action, reward, next_state,
                                 env_embed_4d.cpu().numpy())

                state = hh%100
                state=np.array([state],dtype=np.intc)

            if len(meta_buffer) < 10:
                continue

            # 6) 计算对 meta-params 的损失 (meta-loss)
            meta_data = meta_buffer.sample_all()
            m_actor_loss = compute_mc_actor_loss(actor_fast, meta_data, gamma, device)
            actor_loss_list.append(m_actor_loss)

        if len(actor_loss_list) > 0:
            meta_actor_loss_val = torch.mean(torch.stack(actor_loss_list))
            meta_actor_optim.zero_grad()
            meta_actor_loss_val.backward()
            torch.nn.utils.clip_grad_norm(meta_actor.parameters(), 10)
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

if __name__ == "__main__":
    main()
"""