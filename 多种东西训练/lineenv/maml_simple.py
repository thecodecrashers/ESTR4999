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

# 假设你的自定义文件 lineEnv.py 中定义了 lineEnv
from lineEnv import lineEnv

# ===============================
# 1) 只保留 Actor 网络
#    输出的是一个标量 logit，用于和 λ 作比较
# ===============================
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

        # 自定义初始化：这里演示用常数初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0.01)
            nn.init.constant_(m.bias, 0.01)

    def forward(self, state, env_embed_3d):
        """
        state: (batch, state_dim)
        env_embed_3d: (batch, 3) -> (p, q, OptX)
        return: logit (batch, 1)
        """
        x = torch.cat([state, env_embed_3d], dim=-1)  # (batch, state_dim+3)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        logit = self.fc5(x)
        logit = torch.clamp(logit, min=-2, max=2)  # 限幅可选
        return logit


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, env_embed):
        self.buffer.append((state, action, reward, next_state, env_embed))

    def sample_all(self):
        """
        这里返回全部数据，用于一次性计算蒙特卡洛回报。
        也可改成随机 batch 采样。
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


def clone_model(model: nn.Module):
    import copy
    return copy.deepcopy(model)


# ==================================================
#  新的更新函数：基于长期收益比较来决定如何推高/推低 logit
# ==================================================
def compute_custom_actor_loss(actor, transitions, gamma, device):
    """
    1) 先对整条轨迹做蒙特卡洛回报计算，得到每一步 (state, action) 的回报 G_t。
    2) 按 state+action 分组，计算平均回报 Q_hat(s,a) = E[G_t | s,a]。
    3) 对同一个状态 s 比较 Q_hat(s,1) 和 Q_hat(s,0) 看哪个更大。
       - 若 Q_hat(s,1) > Q_hat(s,0)，则“做”更好。
       - 若 Q_hat(s,0) >= Q_hat(s,1)，则“不做”更好。
    4) 再看当前 actor 给出的动作(通过 logit 与 lam 的比较)是否一致：
       - 如果应该做且当前也做 => 无惩罚
       - 如果应该做但当前没做 => 希望 logit >= lam => hinge = (lam - logit)^2
       - 如果应该不做但当前做了 => 希望 logit < lam => hinge = (logit - lam)^2
       - 如果一致 => loss=0
    """
    # 取出数据
    states_np, actions_np, rewards_np, next_states_np, env_embeds_np = transitions

    batch_size = len(states_np)
    if batch_size == 0:
        # buffer为空，直接返回一个对参数无影响但requires_grad=True的零张量
        # 方式一：返回actor某个参数的和 * 0
        return sum(p.sum() for p in actor.parameters()) * 0.0

    # 转成 torch
    states_t = torch.tensor(states_np, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions_np, dtype=torch.long, device=device)
    rewards_t = torch.tensor(rewards_np, dtype=torch.float32, device=device)
    env_embeds_t = torch.tensor(env_embeds_np, dtype=torch.float32, device=device)

    # ---------- 1) 从后往前折扣累加回报 G_t ----------
    returns_np = np.zeros_like(rewards_np)
    G = 0.0
    for i in reversed(range(batch_size)):
        G = rewards_np[i] + gamma * G
        returns_np[i] = G

    returns_t = torch.tensor(returns_np, dtype=torch.float32, device=device)

    # ---------- 2) 按 (state, action) 分组计算平均回报 ----------
    from collections import defaultdict
    sa_returns = defaultdict(list)  # key=(state_str, action), val=list of returns
    for i, ret_val in enumerate(returns_np):
        # 如果 state_dim=1，这里可以直接用 float(state_np[i][0]) 做key
        # 若 state_dim>1，可用 tuple(state_np[i]) 或 str(state_np[i].tolist())
        s_str = str(states_np[i].tolist())
        a_val = actions_np[i]
        sa_returns[(s_str, a_val)].append(ret_val)

    sa_avg_return = {}
    for k, v in sa_returns.items():
        sa_avg_return[k] = np.mean(v)

    # ============== 3&4) 构建 hinge 损失 ==============
    # 这里要用“可导”的方式计算logit
    logit_full = actor(states_t, env_embeds_t[:, :3]).squeeze(-1)  # shape=[batch]
    lam_full   = env_embeds_t[:, 3]                                # shape=[batch]

    loss_list = []
    for i in range(batch_size):
        # 先确定状态对应的 Q_hat(s,1) 和 Q_hat(s,0)
        s_str = str(states_np[i].tolist())
        q_do  = sa_avg_return.get((s_str, 1), 0.0)  # 若没出现过 a=1，就默认 0
        q_not = sa_avg_return.get((s_str, 0), 0.0)
        do_better = (q_do > q_not)  # bool

        # 期望动作
        want_act = 1 if do_better else 0

        # 当前 actor 给出的动作
        logit_i = logit_full[i]
        lam_i   = lam_full[i]
        agent_act = (logit_i >= lam_i).long()

        if want_act == 1 and agent_act == 0:
            # 应该做但没做 => hinge = (lam - logit)^2
            margin = lam_i - logit_i
            # 只有 margin>0 时才有损失
            loss_list.append(torch.clamp(margin, min=0)**2)

        elif want_act == 0 and agent_act == 1:
            # 应该不做但做了 => hinge = (logit - lam)^2
            margin = logit_i - lam_i
            loss_list.append(torch.clamp(margin, min=0)**2)

        else:
            # 动作正确，无惩罚
            loss_list.append(torch.tensor(0.0, device=device))

    if len(loss_list) == 0:
        # 这里说明batch_size>0，但不知道为什么都没有产生可训练项
        # 返回一个可导的常数0
        return logit_full.sum() * 0.0

    total_loss = torch.mean(torch.stack(loss_list))
    return total_loss


# ====================================
# 主要 MAML + 新的“比较 logit 与 λ” 训练循环
# ====================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- 参数 ----------
    N = 100
    OptX = 99

    # 定义一组任务: (p,q)
    nb_arms = 50
    prob_values = np.linspace(start=0.1, stop=0.9, num=nb_arms)
    # 这里示范 p=q 的场景
    pq_tasks = [(float(p), float(p)) for p in prob_values]
    # lambda 的最小值和最大值
    lambda_min = 0.0
    lambda_max = 2.0

    state_dim = 1  # 每个状态可用 1 维表示

    # MAML 超参数
    meta_iterations = 2000        # 先设小一点，测试运行
    meta_batch_size = 32
    inner_lr = 1e-4
    meta_lr = 1e-4
    gamma = 0.99

    # 每个任务采集多少步用于 adaptation & meta
    adaptation_steps_per_task = 128
    meta_steps_per_task = 128

    # 构建 Meta-Actor
    meta_actor = Actor(state_dim).to(device)
    meta_actor_optim = optim.Adam(meta_actor.parameters(), lr=meta_lr)

    meta_losses_log = []

    for outer_iter in tqdm(range(meta_iterations), desc="Meta Iteration"):
        actor_loss_list = []

        # 1) 从任务集中采样 meta_batch_size 个 (p, q)
        tasks_batch = random.sample(pq_tasks, meta_batch_size)

        for (p_val, q_val) in tasks_batch:
            lam = random.uniform(lambda_min, lambda_max)

            # 构建该任务对应环境
            env = lineEnv(seed=42, N=N, OptX=OptX, p=p_val, q=q_val)

            # env_embed_4d = (p, q, OptX, lam)
            env_embed_4d = torch.tensor([p_val, q_val, float(OptX), lam],
                                        dtype=torch.float32, device=device)

            # 2) 克隆 meta_actor 得到 fast_actor
            actor_fast = clone_model(meta_actor)
            fast_actor_optim = optim.Adam(actor_fast.parameters(), lr=inner_lr)

            # 3) 收集少量数据 (rollout) 做内环 adaptation
            adapt_buffer = ReplayBuffer()
            state = env.reset()
            for _ in range(adaptation_steps_per_task):
                state_arr = np.array(state, dtype=np.float32)

                # forward 得到 logit
                s_t = torch.from_numpy(state_arr).unsqueeze(0).to(device)  # (1,1)
                with torch.no_grad():
                    logit = actor_fast(s_t, env_embed_4d[:3].unsqueeze(0))  # (1,1)
                    lam_val = env_embed_4d[3].item()
                    action = 1 if logit.item() >= lam_val else 0

                # 与环境交互
                next_state, reward, done, _ = env.step(action)
                if action == 1:
                    reward -= lam_val  # 做动作要扣除 lam

                adapt_buffer.push(state_arr, action, reward, next_state,
                                  env_embed_4d.cpu().numpy())

                state = next_state
                if done:
                    state = env.reset()

            # 用 adapt_buffer 计算内环损失并更新 fast_actor
            adapt_data = adapt_buffer.sample_all()
            a_actor_loss = compute_custom_actor_loss(actor_fast, adapt_data, gamma, device)
            # 若 a_actor_loss 不需要 grad，这里可能会报错 => 我们做个检查
            if a_actor_loss.requires_grad:
                fast_actor_optim.zero_grad()
                a_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_fast.parameters(), 10)
                fast_actor_optim.step()

            # 5) 用更新后的 fast_actor 收集 meta 数据
            meta_buffer = ReplayBuffer()
            state = env.reset()
            for _ in range(meta_steps_per_task):
                state_arr = np.array(state, dtype=np.float32)
                s_t = torch.from_numpy(state_arr).unsqueeze(0).to(device)
                with torch.no_grad():
                    logit = actor_fast(s_t, env_embed_4d[:3].unsqueeze(0))
                    lam_val = env_embed_4d[3].item()
                    action = 1 if logit.item() >= lam_val else 0

                next_state, reward, done, _ = env.step(action)
                if action == 1:
                    reward -= lam_val

                meta_buffer.push(state_arr, action, reward, next_state,
                                 env_embed_4d.cpu().numpy())

                state = next_state
                if done:
                    state = env.reset()

            # 6) 计算对 meta-params 的损失 (meta-loss) —— 注意此处用的是 fast_actor
            if len(meta_buffer) > 0:
                meta_data = meta_buffer.sample_all()
                m_actor_loss = compute_custom_actor_loss(actor_fast, meta_data, gamma, device)

                # 只有当 m_actor_loss 跟网络参数连接，才有梯度可回传
                if m_actor_loss.requires_grad:
                    actor_loss_list.append(m_actor_loss)

        if len(actor_loss_list) > 0:
            meta_actor_loss_val = torch.mean(torch.stack(actor_loss_list))
            if meta_actor_loss_val.requires_grad:
                meta_actor_optim.zero_grad()
                meta_actor_loss_val.backward()
                torch.nn.utils.clip_grad_norm_(meta_actor.parameters(), 10)
                meta_actor_optim.step()

            meta_losses_log.append(meta_actor_loss_val.item())
            print(f"[Meta Iter={outer_iter}] loss={meta_actor_loss_val.item():.3f}")
        else:
            # 没有任何可训练的loss，可能是因为所有batch都无需要更新
            meta_losses_log.append(0.0)

    # ============= 画一下 meta-loss 曲线 ============ 
    plt.figure(figsize=(7,5))
    plt.plot(meta_losses_log, label="Meta-Actor Loss (Hinge Update)")
    plt.xlabel("Meta-Iteration")
    plt.ylabel("Loss")
    plt.title("MAML + Compare(Logit, λ) with Hinge-like Update")
    plt.legend()

    # 创建保存目录
    os.makedirs("maml_simple", exist_ok=True)
    plt.savefig("maml_simple/loss_curve.png")
    plt.show()

    # 保存最终模型参数
    torch.save(meta_actor.state_dict(), "maml_simple/meta_actor.pth")


if __name__ == "__main__":
    main()
