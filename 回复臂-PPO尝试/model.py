import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================
# Actor-Critic 模型
# Actor: 输入状态，输出动作概率 (sigmoid for binary action)
# Critic: 输入状态 + λ，输出 V(s, λ)
# ======================================

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
