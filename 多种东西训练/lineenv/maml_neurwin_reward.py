import os
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# 这里要确保你有同级的文件: lineEnv.py 以及 maml_neurwin.py
# 其中, maml_neurwin.py 里已经定义了 Actor、compute_mc_actor_loss 等，
# 并训练出了 meta_actor.pth。

from lineEnv import lineEnv
from maml_neurwin import Actor, compute_mc_actor_loss, ReplayBuffer

# 温度超参数，可与训练时一致
Temperature = 1.0

def clone_model(model: torch.nn.Module):
    """
    简单的深拷贝一个模型。
    """
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
    inner_lr=1e-4,
    K=30
):
    actor_fast = clone_model(meta_actor).to(device)
    fast_actor_optim = optim.Adam(actor_fast.parameters(), lr=inner_lr)

    for _ in range(K):
        lam = random.uniform(lam_min, lam_max)

        # 收集数据
        buffer = ReplayBuffer()
        state = env.reset()
        for _ in range(adaptation_steps):
            state_arr = np.array(state, dtype=np.float32)
            s_t = torch.tensor(state_arr, dtype=torch.float32, device=device).unsqueeze(0)

            # Actor只用 env_embed=(p,q,OptX)，但保存时还要含 lambda
            env_embed_4d = [env.p, env.q, float(env.OptX), lam]

            with torch.no_grad():
                logit = actor_fast(s_t, torch.tensor(env_embed_4d[:3], dtype=torch.float32, device=device).unsqueeze(0))
                p1 = torch.sigmoid((logit - lam) / Temperature).item()
                action = 1 if random.random() < p1 else 0

            next_state, reward, done, _ = env.step(action)
            if action == 1:
                reward -= lam

            buffer.push(state_arr, action, reward, next_state, env_embed_4d)

            state = next_state
            if done:
                state = env.reset()

        # 如果数据不足，跳过更新
        if len(buffer) < 10:
            continue

        # 用采样到的数据更新 fast_actor
        transitions = buffer.sample_all()
        a_loss = compute_mc_actor_loss(actor_fast, transitions, gamma, device)
        fast_actor_optim.zero_grad()
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
                logit = actors_list[i](s_t, env_embed_3d)
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
    """
    绘制所有环境在1-100状态下的logits。
    """
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
                logit = actor_i(state_tensor, env_embed_3d).item()
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
    meta_actor_path = "maml_neurwin/meta_actor.pth"
    meta_actor = Actor(state_dim=1).to(device)
    meta_actor.load_state_dict(torch.load(meta_actor_path, map_location=device))
    meta_actor.eval()

    # ============ 2) 生成若干单臂环境 ============
    num_arms = 10
    N = 100
    OptX = 99
    arms_data = []
    p_vals = np.linspace(start=0.2, stop=0.8, num=num_arms)
    q_vals = np.linspace(start=0.8, stop=0.2, num=num_arms)
    for i in range(num_arms):
        p_val = p_vals[i]
        q_val = q_vals[i]
        seed_val = random.randint(0, 99999)
        arms_data.append((p_val, q_val, seed_val))

    # ============ 3) 对每个臂都执行若干步适应(Adaptation) ============
    # 其中会随机抽取 lambda 并用蒙特卡洛的 REINFORCE 更新
    adaptation_steps = 200
    adapt_inner_lr = 1e-3
    K = 3  # 适应时，重复几次“采样+更新”

    arm_actors = []
    test_envs = []
    for i, (p_val, q_val, seed_val) in enumerate(arms_data):
        env_i = lineEnv(seed=seed_val, N=N, OptX=OptX, p=p_val, q=q_val)

        fast_actor = adapt_single_arm(
            meta_actor=meta_actor,
            env=env_i,
            device=device,
            lam_min=0.0,
            lam_max=2.0,
            adaptation_steps=adaptation_steps,
            gamma=0.99,
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

    out_dir = "maml_neurwin_lin_out"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "test_result_top_k.png"))
    plt.show()

    if len(avgR_plt) > 0:
        print("[Info] Final reported average reward =", avgR_plt[-1])
    else:
        print("[Info] No reported data.")


if __name__ == "__main__":
    main()

