import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from singlearm import SingleArmAoIEnv
from tqdm import tqdm
import random

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(5, 128)  # Inputs: pg, ps, a, d,action
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)  # Outputs: Q-values for actions 0 and 1

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# 初始化十个 QNetwork 实例
q_networks = [QNetwork() for _ in range(10)]

whittle_tables = [np.zeros((11, 11), dtype=np.float64) for _ in range(10)]

discounted = 0.99

with open("probability_pairs_M5_N10.json", "r") as file:
    probability_pairs = json.load(file)

envs = [SingleArmAoIEnv(pg=prob["pg"], ps=prob["ps"]) for prob in probability_pairs]
states = [env.state for env in envs]
print(states)

random.seed(42)

# 使用 tqdm 添加进度条
total_steps = 10000
with tqdm(total=total_steps, desc="Training", unit="step") as pbar:
    for t in range(total_steps):
        prob = 100 / (t + 1000)
        alpha = 1 / (1 + 0.001 * t)
        states = [env.state for env in envs]

        if random.random() < prob:
            actions = random.sample(range(10), 5)
        else:
            values = [whittle_tables[i][state[0], state[1]] for i, state in enumerate(states)]
            sorted_indices = np.argsort(values)[::-1]  # 按降序排列索引
            actions = sorted_indices[:5].tolist()

        arm_actions = [0] * 10
        for arm in actions:
            arm_actions[arm] = 1

        optimizers = [optim.Adam(q_net.parameters(), lr=0.001) for q_net in q_networks]

        for i, env in enumerate(envs):
            next_state, reward, done, _, _ = env.step(arm_actions[i])
            state = states[i]
            action = arm_actions[i]
            pg = probability_pairs[i]["pg"]
            ps = probability_pairs[i]["ps"]

            current_input = torch.tensor([pg, ps, state[0], state[1], action], dtype=torch.float32).unsqueeze(0)

            next_inputs = [
                torch.tensor([pg, ps, next_state[0], next_state[1], a], dtype=torch.float32).unsqueeze(0)
                for a in [0, 1]
            ]

            current_q_value = q_networks[i](current_input)

            # 计算下一状态最大 Q 值
            with torch.no_grad():
                next_q_values = [q_networks[i](next_input) for next_input in next_inputs]
                max_next_q_value = torch.max(torch.stack(next_q_values))

            target_q_value = (reward + discounted * max_next_q_value) * alpha + (1 - alpha) * current_q_value

            optimizers[i].zero_grad()  # 清除梯度
            loss = nn.MSELoss()(current_q_value, target_q_value.unsqueeze(0))  # 计算损失
            loss.backward()  # 反向传播
            optimizers[i].step()

            with torch.no_grad():
                positive = torch.tensor([pg, ps, state[0], state[1], 1], dtype=torch.float32).unsqueeze(0)
                passive = torch.tensor([pg, ps, state[0], state[1], 0], dtype=torch.float32).unsqueeze(0)
                whittle_tables[i][state[0], state[1]] = q_networks[i](positive) - q_networks[i](passive)

        # 更新进度条
        pbar.update(1)
# 补全未访问过的 Whittle indices
for i, whittle_table in enumerate(whittle_tables):
    for a in range(whittle_table.shape[0]):
        for d in range(whittle_table.shape[1]):
            if whittle_table[a, d] == 0:  # 未更新过的值
                pg = probability_pairs[i]["pg"]
                ps = probability_pairs[i]["ps"]
                positive = torch.tensor([pg, ps, a, d, 1], dtype=torch.float32).unsqueeze(0)
                passive = torch.tensor([pg, ps, a, d, 0], dtype=torch.float32).unsqueeze(0)
                whittle_table[a, d] = q_networks[i](positive) - q_networks[i](passive)
# 保存 Whittle indices 到文件
whittle_indices = {}
for i, whittle_table in enumerate(whittle_tables):
    whittle_indices[f"arm_{i}"] = whittle_table.tolist()  # 转换为列表格式以便 JSON 序列化

output_file = "WIQL_N10_M5.json"

with open(output_file, "w") as file:
    json.dump(whittle_indices, file, indent=4)

print(f"Whittle indices saved to: {output_file}")
