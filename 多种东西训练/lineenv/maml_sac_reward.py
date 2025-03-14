import os
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from lineEnv import lineEnv
from maml_sac import Actor, Critic, ReplayBuffer, compute_sac_losses, clone_model

# 温度
Temperature = 1

def adapt_single_arm_with_many_lams(meta_actor,
                                    meta_critic,
                                    env: lineEnv,
                                    device,
                                    lam_min=0.0,
                                    lam_max=2.0,
                                    num_lams=5,
                                    rollout_steps=200,
                                    batch_size=128,
                                    gamma=0.99,
                                    alpha=0,
                                    inner_lr=0.005,
                                    K_outer=3):
    """
    针对同一个单臂环境，进行 K_outer 个外层循环。
    每次循环中：
      1) 随机生成 num_lams 个不同的 lambda
      2) 采集每个 lambda 的 rollout_steps 步数据 (使用 p(a=1)=sigmoid(logit - lambda))
      3) 把这些数据存到 ReplayBuffer
      4) 从 Buffer 采样一次，并对 fast 网络做一次更新
    
    最终返回更新后的 (actor_fast, critic_fast).
    """

    # 1) 克隆 meta 参数 => fast
    actor_fast = clone_model(meta_actor).to(device)
    critic_fast = clone_model(meta_critic).to(device)

    fast_actor_optim = optim.Adam(actor_fast.parameters(), lr=inner_lr)
    fast_critic_optim = optim.Adam(critic_fast.parameters(), lr=inner_lr)
    lam_list=[random.uniform(lam_min, lam_max) for _ in range(K_outer)]
    # 2) 做 K_outer 个外层循环
    for outer_iter in range(K_outer):
        # 2.1) 每次循环都构造一个新的 ReplayBuffer
        lam_val=lam_list[outer_iter]
        ss = np.random.randint(1, 101)  # 这里假设状态是 [1, 100] 的整数
        ss=np.array([ss], dtype=np.intc)
        ss = np.array(ss, dtype=np.float32)
        ss = torch.from_numpy(ss).to(device).unsqueeze(0)  # 转换为 tensor
        with torch.no_grad():
            value = meta_actor(ss, torch.tensor([env.p, env.q, float(99)], dtype=torch.float32, device=device).unsqueeze(0))
            # 确保 value 是一个 Python 数值
        if isinstance(value, torch.Tensor):
            value = value.item()
            # 如果 value 在 lambda 允许的范围内，则替换 lam
        if 0 >value or value> 2:
            lam_val = value
        buffer = ReplayBuffer(max_size=num_lams * rollout_steps + 10)  # buffer 大小够用即可
        state = env.reset()
        for _ in range(rollout_steps):
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            env_embed_3d = torch.tensor([[env.p, env.q, float(env.OptX)]],
                                            dtype=torch.float32, device=device)
            with torch.no_grad():
                logit = actor_fast(s_t, env_embed_3d)
                prob_1 = torch.sigmoid((logit - lam_val)/Temperature).item()
                action = 1 if random.random() < prob_1 else 0

            next_state, reward, done, _ = env.step(action)

                # Critic 需要 (p,q,OptX, lam_val)
            env_embed_4d = [env.p, env.q, float(env.OptX), lam_val]
            buffer.push(state, action, reward, next_state, env_embed_4d)

            state = next_state
            if done:
                state = env.reset()
        # 2.3) 对每个 lam，采集 rollout_steps 步数据
        """for lam_val in lam_list:
            state = env.reset()
            for _ in range(rollout_steps):
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                env_embed_3d = torch.tensor([[env.p, env.q, float(env.OptX)]],
                                            dtype=torch.float32, device=device)
                with torch.no_grad():
                    logit = actor_fast(s_t, env_embed_3d)
                    prob_1 = torch.sigmoid((logit - lam_val)/Temperature).item()
                    action = 1 if random.random() < prob_1 else 0

                next_state, reward, done, _ = env.step(action)

                # Critic 需要 (p,q,OptX, lam_val)
                env_embed_4d = [env.p, env.q, float(env.OptX), lam_val]
                buffer.push(state, action, reward, next_state, env_embed_4d)

                state = next_state
                if done:
                    state = env.reset()"""

        # 2.4) 如果数据不足, 不做更新
        if len(buffer) < batch_size:
            continue

        # 2.5) 从 Buffer 中采样一次并更新
        batch_data = buffer.sample(batch_size)
        a_loss, c_loss = compute_sac_losses(
            actor_fast, critic_fast,
            batch_data,
            gamma=gamma, alpha=alpha,
            device=device
        )
        fast_actor_optim.zero_grad()
        fast_critic_optim.zero_grad()
        a_loss.backward(retain_graph=True)
        c_loss.backward(retain_graph=True)
        fast_actor_optim.step()
        fast_critic_optim.step()

        print(f"[Adapt Outer Loop] Iter={outer_iter+1}/{K_outer}, a_loss={a_loss.item():.4f}, c_loss={c_loss.item():.4f}")

    return actor_fast, critic_fast


def test_multi_arms_top_k(actors_list, envs_list, device,
                          k=3,
                          total_steps=13000,
                          warmup_steps=3000,
                          report_interval=200):
    num_arms = len(envs_list)
    states = [env.reset() for env in envs_list]

    total_reward_after_warmup = 0.0
    steps_after_warmup = 0
    report_records = []  # (global_step, average_reward)

    for step_i in range(1, total_steps + 1):
        # 1) 计算每个臂的 logit
        logits = []
        for i in range(num_arms):
            s_t = torch.tensor(states[i], dtype=torch.float32, device=device).unsqueeze(0)
            env_i = envs_list[i]

            env_embed_3d = torch.tensor([[env_i.p, env_i.q, float(env_i.OptX)]],
                                        dtype=torch.float32, device=device)
            with torch.no_grad():
                logit = actors_list[i](s_t, env_embed_3d)
                logits.append(logit.item())

        # 2) 找出 logit 最大的 k 个臂 => action=1，其余=0
        chosen_indices = np.argsort(logits)[-k:]
        chosen_indices = set(chosen_indices)  # 方便判断

        # 3) 执行动作并获得回报
        step_reward_sum = 0.0
        for i in range(num_arms):
            action = 1 if i in chosen_indices else 0
            next_state, reward, done, _ = envs_list[i].step(action)
            step_reward_sum += reward
            states[i] = next_state
            if done:
                states[i] = envs_list[i].reset()

        # 4) 统计回报（忽略前 warmup_steps 步）
        if step_i > warmup_steps:
            total_reward_after_warmup += step_reward_sum
            steps_after_warmup += 1

            if (step_i % report_interval) == 0:
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
    print("[Info] Using device:", device)

    # ============ 1) 加载已有的 meta-Actor / meta-Critic ============
    meta_actor = Actor(state_dim=1).to(device)
    meta_critic = Critic(state_dim=1).to(device)

    meta_actor.load_state_dict(torch.load("maml_sac/meta_actor.pth", map_location=device))
    meta_critic.load_state_dict(torch.load("maml_sac/meta_critic.pth", map_location=device))

    # ============ 2) 生成多个单臂环境 ============
    num_arms = 10
    N = 100
    OptX = 99
    arms_data = []
    p = np.linspace(start=0.2, stop=0.8, num=num_arms)
    q = p
    for i in range(num_arms):
        p_val = p[i]
        q_val = p[i]
        seed_val = 42
        arms_data.append((p_val, q_val, seed_val))

    # ============ 3) 对每个臂都随机多条 lambda，做 K_outer 次外层循环采集和更新 ============
    num_lams = 15
    lam_min = 0.0
    lam_max = 2.0
    rollout_steps = 200

    # 这里就是您想要的多次外层循环次数
    K_outer = 1

    arm_actors = []
    arm_critics = []
    for i in range(num_arms):
        p_val, q_val, seed_val = arms_data[i]
        env_i = lineEnv(seed=seed_val, N=N, OptX=OptX, p=p_val, q=q_val)

        actor_fast_i, critic_fast_i = adapt_single_arm_with_many_lams(
            meta_actor=meta_actor,
            meta_critic=meta_critic,
            env=env_i,
            device=device,
            lam_min=lam_min,
            lam_max=lam_max,
            num_lams=num_lams,
            rollout_steps=rollout_steps,
            batch_size=64,
            gamma=0.99,
            alpha=0,
            inner_lr=0.0001,
            K_outer=K_outer
        )
        arm_actors.append(actor_fast_i)
        arm_critics.append(critic_fast_i)

        print(f"[Arm {i}] p={p_val:.3f}, q={q_val:.3f}, seed={seed_val} => done multi-lam adaptation with K_outer={K_outer}.")

    print("[Info] Finished adaptation for all arms.")

    # ============ 4) 并行测试 (Top-3 决策，不再用 lambda) ============
    test_envs = []
    for i in range(num_arms):
        p_val, q_val, seed_val = arms_data[i]
        test_envs.append(lineEnv(seed=seed_val, N=N, OptX=OptX, p=p_val, q=q_val))

    total_steps_test = 13000
    warmup_steps = 3000
    report_interval = 200
    top_k = 3
    plot_logits_across_states(arm_actors, test_envs, device)
    report_records = test_multi_arms_top_k(
        arm_actors,
        test_envs,
        device=device,
        k=top_k,
        total_steps=total_steps_test,
        warmup_steps=warmup_steps,
        report_interval=report_interval
    )

    # ============ 5) 画图输出 ============
    steps_plt = [r[0] for r in report_records]
    avgR_plt = [r[1] for r in report_records]

    plt.figure(figsize=(6, 5))
    plt.plot(steps_plt, avgR_plt, marker='o', label="Avg Reward (Top-3 selection, from step>3000)")
    plt.xlabel("Global Step")
    plt.ylabel("Average Reward")
    plt.title("Multi-Arm parallel test (K_outer repeated data collection & update)")
    plt.legend()

    os.makedirs("maml_lineenv_out", exist_ok=True)
    plt.savefig("maml_lineenv_out/test_result_top3_many_lams_K_outer.png")
    plt.show()

    if len(avgR_plt) > 0:
        print("[Info] Final reported average reward =", avgR_plt[-1])
    else:
        print("[Info] No reported data.")


if __name__ == "__main__":
    main()

