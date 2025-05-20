import os
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from collections import deque
from lineEnv import lineEnv
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
    for i in reversed(range(batch_size)):
        G = rewards[i] + gamma * G
        returns[i] = G
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    logit = actor(torch.cat([states_t, env_embeds_t[:, :3]], dim=-1))  # (batch,1)
    lam_val = env_embeds_t[:, 3:4]  # (batch,1)
    Temperature = 1.0  # 你需要定义 Temperature
    logit_adj = (logit - lam_val) / Temperature  # (batch,1)
    p1 = torch.sigmoid(logit_adj)  # (batch,1)
    probs = torch.cat([1 - p1, p1], dim=-1)  # (batch,2)
    log_probs = torch.log(probs + 1e-8)  # (batch,2)
    chosen_log_probs = log_probs.gather(1, actions_t.view(-1, 1)).squeeze(-1)  # (batch,)
    actor_loss = - torch.mean(returns_t[:half_batch] * chosen_log_probs[:half_batch])
    return actor_loss
"""def compute_mc_actor_loss(actor, transitions, gamma, device):
    states, actions, rewards, _, env_embeds = transitions
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
    logit = actor(torch.cat([states_t, env_embeds_t[:, :3]],dim=-1))  # (batch,1)
    lam_val = env_embeds_t[:, 3:4]                # (batch,1)
    # 这里做 (logit - lam) / Temperature
    logit_adj = (logit - lam_val) / Temperature   # (batch,1)
    p1 = torch.sigmoid(logit_adj)                 # (batch,1)

    # 拼成对 [p(a=0), p(a=1)]
    probs = torch.cat([1 - p1, p1], dim=-1)       # (batch,2)
    log_probs = torch.log(probs + 1e-8)           # (batch,2)

    # 取对应action的 log_prob
    chosen_log_probs = log_probs.gather(1, actions_t.view(-1,1)).squeeze(-1)  # (batch,)
    actor_loss = - torch.mean(returns_t * chosen_log_probs)
    return actor_loss"""


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
        #self.activation=nn.SELU()
        #self.activation=nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, -0.00001)
            nn.init.constant_(m.bias, -0.00001)

    def forward(self, state_env):
        x = self.fc1(state_env)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        logit = self.fc5(x)  
        return logit
# 温度超参数，可与训练时一致
Temperature = 1.0

def clone_model(model: torch.nn.Module):
    import copy
    return copy.deepcopy(model)


def adapt_single_arm(
    meta_actor,
    env: lineEnv,
    device,
    lam_min=0.0,
    lam_max=2.0,
    adaptation_steps=200,
    gamma=0.99,
    inner_lr=1e-3,
    K=30
):
    actor_fast = clone_model(meta_actor).to(device)
    fast_actor_optim = optim.SGD(actor_fast.parameters(), lr=inner_lr)
    buffer = ReplayBuffer()
    for _ in range(K):
        lam = random.uniform(lam_min, lam_max)       
#        buffer = ReplayBuffer()
        state = env.reset()
        value=random.randint(0,100)
        state=np.array([value], dtype=np.intc)
        for _ in range(adaptation_steps):
            state_arr = np.array(state, dtype=np.float32)
            s_t = torch.tensor(state_arr, dtype=torch.float32, device=device).unsqueeze(0)
            env_embed_4d = [env.p, env.q, float(env.OptX), lam]
            with torch.no_grad():
                logit = actor_fast(torch.cat([s_t, torch.tensor(env_embed_4d[:3], dtype=torch.float32, device=device).unsqueeze(0)],dim=-1))
                p1 = torch.sigmoid((logit - lam) / Temperature).item()
                #action = 1 if random.random() < p1 else 0
                action =0 if random.random() > p1 or p1<0.5 else 1
            next_state, reward, done, _ = env.step(action)
            if action == 1:
                reward -= lam
            buffer.push(state_arr, action, reward, next_state, env_embed_4d)
            state = next_state
            if done:
                state = env.reset()
        if len(buffer) < 10:
            continue
        transitions = buffer.sample_all()
        a_loss = compute_mc_actor_loss(actor_fast, transitions, gamma, device)
        fast_actor_optim.zero_grad()
        torch.nn.utils.clip_grad_norm(actor_fast.parameters(), max_norm=1.0)
        a_loss.backward()
        fast_actor_optim.step()
    return actor_fast


def test_multi_arms_top_k(actors_list, envs_list, device,
                          k=3,
                          total_steps=10000,
                          warmup_steps=2000,
                          report_interval=200):
    num_arms = len(envs_list)
    states = [env.reset() for env in envs_list]

    total_reward_after_warmup = 0.0
    steps_after_warmup = 0
    report_records = []

    for step_i in range(1, total_steps + 1):
        # 1) 计算每个臂的 logit
        logits = []
        for i in range(num_arms):
            s_t = torch.tensor(states[i], dtype=torch.float32, device=device).unsqueeze(0)
            env_i = envs_list[i]
            # 只用 (p,q,OptX) => env_embed_3d
            env_embed_3d = torch.tensor([env_i.p, env_i.q, float(env_i.OptX)],
                                        dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                logit = actors_list[i](torch.cat([s_t, env_embed_3d],dim=-1))
                logits.append(logit.item())

        # 2) 找出 logit 最大的 k 个臂 => action=1，其余=0
        chosen_indices = np.argsort(logits)[-k:]
        chosen_set = set(chosen_indices)

        # 3) 执行动作并获得回报
        step_reward_sum = 0.0
        for i in range(num_arms):
            action = 1 if i in chosen_set else 0
            next_state, reward, done, _ = envs_list[i].step(action)
            step_reward_sum += reward
            states[i] = next_state
            if done:
                states[i] = envs_list[i].reset()

        # 4) 只在 warmup_steps 之后开始统计
        if step_i > warmup_steps:
            total_reward_after_warmup += step_reward_sum
            steps_after_warmup += 1

            if step_i % report_interval == 0:
                avg_r = total_reward_after_warmup / steps_after_warmup
                report_records.append((step_i, avg_r))

    return report_records

def plot_logits_across_states(actors_list, envs_list, device):
    num_arms = len(envs_list)
    states_range = np.arange(1, 101)
    logits_dict = {}

    plt.figure(figsize=(10, 6))
    
    for i in range(num_arms):
        logits = []
        env_i = envs_list[i]
        actor_i = actors_list[i]
        
        env_embed_3d = torch.tensor([env_i.p, env_i.q, float(env_i.OptX)],
                                    dtype=torch.float32, device=device).unsqueeze(0)
        
        for state_val in states_range:
            state_tensor = torch.tensor([state_val], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logit = actor_i(torch.cat([state_tensor, env_embed_3d],dim=-1)).item()
                logits.append(logit)
        
        logits_dict[f"Arm {i}"] = logits
        plt.plot(states_range, logits, label=f"Arm {i}")
    
    plt.xlabel("State (1-100)")
    plt.ylabel("Logit Value")
    plt.title("Logits for Each Environment Across States")
    plt.legend()
    
    out_dir = "maml_neurwin_lin_out"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "logits_across_states.png"))
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ============ 1) 加载已经训练好的 meta-Actor ============
    # 这是您在 maml_neurwin.py 训练后保存的模型文件
    meta_actor_path = "whittle_model.pth"
    meta_actor = Actor(state_dim=1).to(device)
    meta_actor.load_state_dict(torch.load(meta_actor_path, map_location=device))
    meta_actor.eval()

    # ============ 2) 生成若干单臂环境 ============
    num_arms = 10
    N = 100
    OptX = 99
    arms_data = []
    p_vals = np.linspace(start=0.2, stop=0.8, num=num_arms)
    q_vals = p_vals
    for i in range(num_arms):
        p_val = p_vals[i]
        q_val = q_vals[i]
        seed_val = 42
        arms_data.append((p_val, q_val, seed_val))
    adaptation_steps = 3000
    adapt_inner_lr = 1e-5
    K = 0

    arm_actors = []
    test_envs = []
    for i, (p_val, q_val, seed_val) in enumerate(arms_data):
        env_i = lineEnv(seed=42, N=N, OptX=OptX, p=p_val, q=q_val)

        fast_actor = adapt_single_arm(
            meta_actor=meta_actor,
            env=env_i,
            device=device,
            lam_min=0.0,
            lam_max=2.0,
            adaptation_steps=adaptation_steps,
            gamma=0.9,
            inner_lr=adapt_inner_lr,
            K=K
        )
        arm_actors.append(fast_actor)
        test_envs.append(lineEnv(seed=seed_val, N=N, OptX=OptX, p=p_val, q=q_val))
        print(f"[Arm {i}] p={p_val:.3f}, q={q_val:.3f}, seed={seed_val} => Adaptation done.")

    # ============ 4) 并行测试，统计长期平均回报 ============
    # 比如选取 top-k=3 个臂动作=1，其余=0
    total_steps_test = 10000
    warmup_steps = 2000
    report_interval = 200
    top_k = 3
    plot_logits_across_states(arm_actors, test_envs, device)
    results = test_multi_arms_top_k(
        actors_list=arm_actors,
        envs_list=test_envs,
        device=device,
        k=top_k,
        total_steps=total_steps_test,
        warmup_steps=warmup_steps,
        report_interval=report_interval
    )

    # ============ 5) 画图输出 ============
    steps_plt = [r[0] for r in results]
    avgR_plt = [r[1] for r in results]
    plt.figure(figsize=(7,5))
    plt.plot(steps_plt, avgR_plt, marker='o', label=f"Top-{top_k} selection average reward")
    plt.xlabel("Global Step")
    plt.ylabel("Average Reward")
    plt.title("Test MAML-Neurwin Multi-Arms (Top-k = {})".format(top_k))
    plt.legend()

    out_dir = "my_old_neurwin"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "test_result_top_k.png"))
    plt.show()

    if len(avgR_plt) > 0:
        print("[Info] Final reported average reward =", avgR_plt[-1])
    else:
        print("[Info] No reported data.")


if __name__ == "__main__":
    main()