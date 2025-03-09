import torch
import torch.nn as nn

# 定义 Actor 模型
class Actor(nn.Module):
    def __init__(self, state_dim):
        super(Actor, self).__init__()
        self.input_dim = state_dim + 3
        hidden_dim = 1024  # 增大隐藏层维度
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)  # single logit
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, state, env_embed_3d):
        x = torch.cat([state, env_embed_3d], dim=-1)  # (batch_size, state_dim+3)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        logit = self.fc5(x)  # (batch_size,1)
        return logit

# 加载模型
model_path = 'maml_sac/meta_actor.pth'
state_dim = 1
actor = Actor(state_dim)
actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
actor.eval()  # 评估模式

# 生成 state 1-100 (batch_size=100)
state = torch.tensor([[i] for i in range(1, 101)], dtype=torch.float32)  # (100, 1)
env_embed_3d = torch.tensor([[0.2, 0.2, 99]] * 100, dtype=torch.float32)  # (100, 3)

# 批量推理
with torch.no_grad():
    outputs = actor(state, env_embed_3d)

# 打印所有结果
for i, output in enumerate(outputs):
    print(f"State: {i+1}, Output: {output.item()}")
