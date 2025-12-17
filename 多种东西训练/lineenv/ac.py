import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from lineEnv import lineEnv

# ========= 网络结构 =========
class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)  # 输出 Whittle Index 的估计值

        # 激活函数
        self.activation = nn.ReLU()

        # 初始化最后一层偏置为 5（让初始 index 靠近 λ 平均值）
        #nn.init.constant_(self.output_layer.bias, 5.0)

    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        out = self.output_layer(x)
        return out 

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + 1, hidden_dim)  # 拼接 λ 到 state
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)     # 输出 scalar V(s, λ)

        self.activation = nn.ReLU()

    def forward(self, state, lambd):
        x = torch.cat([state, lambd], dim=1)  # 拼接 λ
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        value = self.output_layer(x)
        return value


# 可选的打包成模块
class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)

    def act(self, state):
        return self.actor(state)

    def value(self, state, lambd):
        return self.critic(state, lambd)

# ========= 超参数 =========
GAMMA = 0.99
GAE_LAMBDA = 0.95
TOTAL_EPISODES = 200000
MAX_STEPS_PER_EPISODE = 100

LAMBDA_MIN, LAMBDA_MAX = -1,2
NUM_LAMBDAS = 10

ACTOR_LR = 1e-4
CRITIC_LR = 1e-5
N_CRITIC_UPDATES = 10
ENTROPY_COEF = 0.01
GRAD_CLIP_NORM = 5.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= GAE =========
def compute_gae(rewards, values, next_value, dones):
    values = values + [next_value]
    gae, returns = 0.0, []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
        returns.insert(0, gae + values[t])
    return returns

# ========= 主训练函数 =========
def train():
    env = lineEnv(seed=42, N=100, OptX=99, p=0.8, q=0.8)
    ac = ActorCritic(state_dim=1).to(device)
    actor_opt = optim.Adam(ac.actor.parameters(), lr=ACTOR_LR)
    critic_opt = optim.Adam(ac.critic.parameters(), lr=CRITIC_LR)

    lambda_bins = np.linspace(LAMBDA_MIN, LAMBDA_MAX, NUM_LAMBDAS + 1)
    tqdm_bar = tqdm(range(TOTAL_EPISODES))

    for ep in tqdm_bar:
        all_s, all_a, all_r, all_v = [], [], [], []
        all_logp, all_ent, all_lam, all_adv = [], [], [], []

        for lam_idx in range(NUM_LAMBDAS):
            lam_val = np.random.uniform(lambda_bins[lam_idx], lambda_bins[lam_idx+1])
            lam_tensor = torch.tensor([[lam_val]], dtype=torch.float32, device=device)

            state = env.reset()
            done, step = False, 0
            states, actions, rewards, values, logps, ents, dones = [], [], [], [], [], [], []

            while step < MAX_STEPS_PER_EPISODE:
                step += 1
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    idx = ac.act(s_t)
                    logits = idx - lam_tensor
                    dist = torch.distributions.Bernoulli(logits=logits)
                    epsilon=0.2
                    if torch.rand(1).item() < epsilon:
                        a_t = torch.bernoulli(torch.full_like(logits, 0.5))
                    else:
                        a_t = dist.sample()
                    a_t = a_t.view(1, 1)
                    #logp = dist.log_prob(a_t)
                    v_t = ac.value(s_t, lam_tensor)

                next_state, reward, done, _ = env.step(int(a_t.item()))
                if int(a_t.item()) == 1:
                    reward -= lam_val

                states.append(s_t)
                actions.append(a_t)
                rewards.append(reward)
                values.append(v_t.item())
                #logps.append(logp)
                dones.append(done)
                #print(state,next_state)
                state = next_state
                if done:
                    break

            next_v = ac.value(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0), lam_tensor).item()
            returns = compute_gae(rewards, values, next_v, dones)
            adv = torch.tensor(returns, device=device) - torch.tensor(values, device=device)

            all_s += states
            all_a += actions
            all_r += returns
            all_v += values
            #all_logp += logps
            all_lam += [lam_val] * len(states)
            all_adv += adv.tolist()

        S = torch.cat(all_s)
        #A = torch.stack(all_a)
        RET = torch.tensor(all_r, dtype=torch.float32, device=device).unsqueeze(1)
        #V = torch.tensor(all_v, dtype=torch.float32, device=device).unsqueeze(1)
        ADV = torch.tensor(all_adv, dtype=torch.float32, device=device).unsqueeze(1)
        ADV = (ADV - ADV.mean()) / (ADV.std(unbiased=False) + 1e-8)
        LAM = torch.tensor(all_lam, dtype=torch.float32, device=device).unsqueeze(1)

        # ======== Critic 更新 ========
        for _ in range(N_CRITIC_UPDATES):
            critic_opt.zero_grad()
            v_pred = ac.value(S, LAM)
            v_loss = ((v_pred - RET) ** 2).mean()
            v_loss.backward()
            nn.utils.clip_grad_norm_(ac.critic.parameters(), GRAD_CLIP_NORM)
            critic_opt.step()

        # ======== Actor 构造 BCE target ========
        with torch.no_grad():
            reward1, reward0 = [], []
            for i in range(len(S)):
                x = int(S[i].item())
                lam = float(LAM[i].item())
                r1, r0 = env._calRewardAndState(1), env._calRewardAndState(0)
                s1 = torch.tensor(r1[0], dtype=torch.float32, device=device).unsqueeze(0)
                s0 = torch.tensor(r0[0], dtype=torch.float32, device=device).unsqueeze(0)
                v1 = r1[1] + GAMMA * ac.value(s1, torch.tensor([[lam]], device=device)).item() - lam
                v0 = r0[1] + GAMMA * ac.value(s0, torch.tensor([[lam]], device=device)).item()
                reward1.append(v1)
                reward0.append(v0)
            target = (np.array(reward1) > np.array(reward0)).astype(np.float32)
            target = torch.tensor(target, device=device).unsqueeze(1)

        # # ======== Actor 更新（带 Mask）========
        # actor_opt.zero_grad()
        # idx_pred = ac.act(S)
        # logits   = (idx_pred - LAM)  # shape: (N,1)

        # # 构造 BCE Loss，但跳过预测正确的点：
        # with torch.no_grad():
        #     # 判断哪些样本“已经是正确方向”
        #     correct = ((target == 1) & (logits > 0)) | ((target == 0) & (logits < 0))  # shape: (N,1)
        #     mask = ~correct  # 保留需要学习的位置

        # # 构造损失
        # bce_all = nn.BCEWithLogitsLoss(reduction='none')(logits, target)  # (N,1)
        # masked_loss = bce_all * mask.float()  # 只保留错的部分
        # if mask.sum() > 0:
        #     loss = masked_loss.sum() / mask.sum()
        #     loss.backward()
        #     nn.utils.clip_grad_norm_(ac.actor.parameters(), GRAD_CLIP_NORM)
        #     actor_opt.step()
        # else:
        #     print(f"[Skip] No actor update at ep {ep} (all correct)")
        #     continue

        actor_opt.zero_grad()
        idx_pred = ac.act(S)
        logits = (idx_pred - LAM) # 放大 logits 以提高训练效果
        bce_loss = nn.BCEWithLogitsLoss()(logits, target)
        loss = bce_loss
        loss.backward()
        nn.utils.clip_grad_norm_(ac.actor.parameters(), GRAD_CLIP_NORM)
        actor_opt.step()

        tqdm_bar.set_description(
            f"Ep {ep+1}/{TOTAL_EPISODES} | Aloss {loss.item():.4f} | Vloss {v_loss.item():.4f}"
        )

    return ac

# ========= 可视化 =========
def visualize_index(ac, max_state=100):
    ac.actor.eval()
    states = torch.arange(0, max_state + 1, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        index_preds = ac.actor(states)
    plt.plot(states.cpu().numpy(), index_preds.cpu().numpy(), label="Whittle Index Estimate")
    plt.xlabel("State")
    plt.ylabel("Index")
    plt.grid(True)
    plt.title("Actor Output vs. State")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ========= 主程序入口 =========
if __name__ == "__main__":
    model = train()
    visualize_index(model)
