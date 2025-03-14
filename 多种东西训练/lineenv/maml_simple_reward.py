import os
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------------------------------------
# 假设你有下列文件:
#   lineEnv.py (定义了 lineEnv)
#   maml_neurwin.py (定义了 Actor, ReplayBuffer 等)
# 并训练出了 meta_actor.pth 存在: maml_neurwin/meta_actor.pth
# --------------------------------------
from lineEnv import lineEnv
from maml_neurwin import Actor, ReplayBuffer


def clone_model(model: torch.nn.Module):
    """
    简单的深拷贝一个模型，用于 fast adaptation.
    """
    import copy
    return copy.deepcopy(model)


def compute_hinge_loss(actor, transitions, device):
    """
    根据“(s, a, r, next_s, env_embed)”批量数据，用 hinge-like 方法更新:
      - 不再使用 sigmoid + REINFORCE
      - 只要决策错误，就产生一个 >0 的 hinge-loss，推动 logit 往正确方向移动
      - 否则 loss=0

    这里为了简化：
      best_action = 1 if state < OptX else 0
    你可根据需求自己判定最优动作(比如真要对比贴现回报或别的逻辑)。
    
    hinge 公式:
      if best=1 => loss_i = max(0, lam - logit_i)
      if best=0 => loss_i = max(0, logit_i - lam)
    """
    states, actions, rewards, next_states, env_embeds = transitions

    # 转为tensor
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    env_embeds_t = torch.tensor(env_embeds, dtype=torch.float32, device=device)
    lam_vals = env_embeds_t[:, 3]   # shape (batch,)

    # 只取 (p,q,OptX) 三维 => actor 输入
    env_embed_3d = env_embeds_t[:, :3]
    
    # forward
    logits = actor(states_t, env_embed_3d).view(-1)  # shape (batch,)

    # 判定最佳动作: best=1 if state<OptX else 0
    #   注意 state 维度可能是 (B,1)，这里先 squeeze
    states_squeezed = states_t.view(-1)
    opt_x_vals = env_embed_3d[:, 2]  # 第2维是 OptX
    best_actions = torch.where(states_squeezed < opt_x_vals, 1, 0)  # shape=(batch,)

    # hinge-like 计算
    # best=1 => hinge = relu( lam - logit )
    # best=0 => hinge = relu( logit - lam )
    diff_1 = lam_vals - logits
    diff_0 = logits - lam_vals

    # 用 (best_actions == 1) 做掩码
    #   best=1时只让 loss = relu(diff_1)，best=0时只让 loss = relu(diff_0)
    mask_1 = (best_actions == 1).float()
    mask_0 = 1.0 - mask_1
    loss_1 = torch.nn.functional.relu(diff_1) * mask_1
    loss_0 = torch.nn.functional.relu(diff_0) * mask_0

    total_loss = loss_1 + loss_0
    return total_loss.mean()


def adapt_single_arm(
    meta_actor,
    env: lineEnv,
    device,
    lam_min=0.0,
    lam_max=2.0,
    adaptation_steps=200,
    inner_lr=1e-4,
    K=30
):
    """
    对单臂环境执行若干次适应:
      1) 克隆 meta_actor => actor_fast
      2) 在每一轮, 随机采样 lam, 采集 adaptation_steps 个样本, 用 hinge-loss 更新
      3) 重复 K 轮
    
    返回适应后的 fast-actor
    """
    # 1) 克隆
    actor_fast = clone_model(meta_actor).to(device)
    fast_actor_optim = optim.Adam(actor_fast.parameters(), lr=inner_lr)

    # 2) 重复 K 轮
    for _ in range(K):
        lam = random.uniform(lam_min, lam_max)

        buffer = ReplayBuffer()
        state = env.reset()

        for _ in range(adaptation_steps):
            # 动作选择: logit vs lam
            state_arr = np.array(state, dtype=np.float32)
            s_t = torch.tensor(state_arr, dtype=torch.float32, device=device).unsqueeze(0)

            # env_embed_4d=(p,q,OptX, lam)，只输入前三维给actor
            env_embed_4d = [env.p, env.q, float(env.OptX), lam]
            env_embed_3d = torch.tensor(env_embed_4d[:3], dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                logit = actor_fast(s_t, env_embed_3d).item()

            if logit > lam:
                action = 1
            else:
                action = 0

            # 与环境交互
            next_state, base_reward, done, _ = env.step(action)
            # 若 action=1 => 减去 lam
            if action == 1:
                base_reward -= lam

            # 存到buffer
            buffer.push(state_arr, action, base_reward, next_state, env_embed_4d)

            state = next_state
            if done:
                state = env.reset()

        # 用 hinge-loss 进行一次更新
        if len(buffer) > 10:
            transitions = buffer.sample_all()
            loss = compute_hinge_loss(actor_fast, transitions, device)
            fast_actor_optim.zero_grad()
            loss.backward()
            fast_actor_optim.step()

    return actor_fast


def test_multi_arms_top_k(actors_list, envs_list, device,
                          k=3,
                          total_steps=10000,
                          warmup_steps=2000,
                          report_interval=200):
    """
    并行测试多臂, 每个臂都由自己的 actor 来决策 logit.
    每个 step:
      1) 算出每个臂的 logit
      2) 找出 logit 最大的 k 个臂 => action=1, 其他action=0
      3) 执行并统计总回报
      4) 只在 warmup_steps 后开始统计平均值
    """
    num_arms = len(envs_list)
    states = [env.reset() for env in envs_list]

    total_reward_after_warmup = 0.0
    steps_after_warmup = 0
    report_records = []

    for step_i in range(1, total_steps + 1):
        # 1) 对每个臂算 logit
        logits = []
        for i in range(num_arms):
            s_t = torch.tensor(states[i], dtype=torch.float32, device=device).unsqueeze(0)
            env_i = envs_list[i]
            env_embed_3d = torch.tensor([env_i.p, env_i.q, float(env_i.OptX)],
                                        dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logit = actors_list[i](s_t, env_embed_3d)
                logits.append(logit.item())

        # 2) 找出 logit 最大的 k 个 => action=1，其余=0
        chosen_indices = np.argsort(logits)[-k:]
        chosen_set = set(chosen_indices)

        # 3) 执行动作并获得回报 (这里不会再做 hinge更新, 纯测试)
        step_reward_sum = 0.0
        for i in range(num_arms):
            action = 1 if i in chosen_set else 0
            next_state, reward, done, _ = envs_list[i].step(action)
            step_reward_sum += reward
            states[i] = next_state
            if done:
                states[i] = envs_list[i].reset()

        # 4) warmup后统计
        if step_i > warmup_steps:
            total_reward_after_warmup += step_reward_sum
            steps_after_warmup += 1

            if step_i % report_interval == 0:
                avg_r = total_reward_after_warmup / steps_after_warmup
                report_records.append((step_i, avg_r))

    return report_records


def plot_logits_across_states(actors_list, envs_list, device):
    """
    绘制所有环境在 1~100 范围状态下的 logit 变化情况 (用于可视化).
    """
    num_arms = len(envs_list)
    states_range = np.arange(1, 101)

    plt.figure(figsize=(10, 6))
    
    for i in range(num_arms):
        logits_list = []
        env_i = envs_list[i]
        actor_i = actors_list[i]
        
        env_embed_3d = torch.tensor([env_i.p, env_i.q, float(env_i.OptX)],
                                    dtype=torch.float32, device=device).unsqueeze(0)
        
        for state_val in states_range:
            s_t = torch.tensor([state_val], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logit_val = actor_i(s_t, env_embed_3d).item()
                logits_list.append(logit_val)
        
        plt.plot(states_range, logits_list, label=f"Arm {i}")
    
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
    # 请确保在 maml_neurwin.py 中已经训练好并保存 meta_actor.pth
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

    # ============ 3) 对每个臂都执行若干步适应 (Hinge-Loss) ============
    adaptation_steps = 200
    adapt_inner_lr = 1e-3
    K = 30

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
            inner_lr=adapt_inner_lr,
            K=K
        )
        arm_actors.append(fast_actor)

        # 用相同参数构造一个专门测试的环境(初始 reset 独立)
        test_envs.append(lineEnv(seed=seed_val, N=N, OptX=OptX, p=p_val, q=q_val))
        print(f"[Arm {i}] p={p_val:.3f}, q={q_val:.3f}, seed={seed_val} => Adaptation done.")

    # ============ 4) 并行测试，统计长期平均回报 ============
    # 选取 top-k=3 臂执行 action=1，其余=0
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
    plt.title(f"Test MAML-Neurwin Multi-Arms (Top-k = {top_k})")
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
