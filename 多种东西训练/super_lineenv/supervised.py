import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from lineEnv import lineEnv

# Define the dataset class
class WhittleDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r") as file:
            data = json.load(file)
        
        self.states = np.array([d["state"] for d in data], dtype=np.float32).reshape(-1, 1)
        self.p_values = np.array([d["p"] for d in data], dtype=np.float32).reshape(-1, 1)
        self.q_values = np.array([d["q"] for d in data], dtype=np.float32).reshape(-1, 1)
        self.optx_values = np.array([d["OptX"] for d in data], dtype=np.float32).reshape(-1, 1)
        self.whittle_values = np.array([d["whittle"] for d in data], dtype=np.float32).reshape(-1, 1)
        
        self.inputs = np.hstack((self.states, self.p_values, self.q_values, self.optx_values))

    def __len__(self):
        return len(self.whittle_values)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.whittle_values[idx])

# Define the Actor network
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

class PolyActor(nn.Module):
    def __init__(self, state_dim):
        super(PolyActor, self).__init__()
        self.input_dim = state_dim + 3  # 原始特征: state + (p, q, OptX)
        self.poly_dim = self.input_dim + 2  # 额外增加 state² 和 state³
        hidden_dim = 128

        self.fc1 = nn.Linear(self.poly_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)  # single logit

        self.activation = nn.GELU()
        #self.activation = nn.SELU()
        #self.activation = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, -0.01)
            nn.init.constant_(m.bias, -0.01)

    def forward(self, state_env):
        # state 是第 0 维, 提取并计算高阶特征
        state = state_env[:, 0:1]  # 提取 state (保持维度)
        state_sq = state ** 2  # 二次项
        state_cube = state ** 3  # 三次项

        # 拼接多项式特征
        poly_features = torch.cat([state_env, state_sq, state_cube], dim=1)

        # 继续前向传播
        x = self.fc1(poly_features)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        logit = self.fc5(x)
        
        return logit
class ResidualActor(nn.Module):
    def __init__(self, state_dim):
        super(ResidualActor, self).__init__()
        self.input_dim = state_dim + 3
        hidden_dim = 256

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

        self.activation = nn.ReLU()  # 用 ReLU 代替 GELU
        self.shortcut = nn.Linear(self.input_dim, 1)  # 直接映射 state

    def forward(self, state_env):
        x = self.fc1(state_env)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)

        out = self.fc_out(x) + self.shortcut(state_env)  # 残差连接
        return out

class TransformerActor(nn.Module):
    def __init__(self, state_dim, num_layers=2, num_heads=4, hidden_dim=256, ff_dim=512, dropout=0.1):
        super(TransformerActor, self).__init__()
        self.input_dim = state_dim + 3  # state + (p, q, OptX)

        # 输入投影
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state_env):
        x = self.input_proj(state_env)  # 线性投影到隐藏维度
        x = x.unsqueeze(1)  # Transformer 需要 (batch, seq_len, hidden_dim)

        x = self.transformer_encoder(x)  # Transformer 编码
        x = x.squeeze(1)  # 去掉 seq 维度

        out = self.fc_out(x)  # 输出 Whittle index 预测值
        return out
# Training the model
def train_whittle_network(json_file, num_epochs=50, batch_size=5000, learning_rate=0.001, model_save_path="whittle_model.pth"):
    dataset = WhittleDataset(json_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = Actor(state_dim=1) 
    #model=ResidualActor(state_dim=1)
    #model=TransformerActor(state_dim=1)
    #model=PolyActor(state_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for state_env, whittles in dataloader:
            optimizer.zero_grad()
            predictions = model(state_env)
            loss = criterion(predictions, whittles)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.6f}")
    
    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model

if __name__ == "__main__":
    json_file = "whittle_indices.json"
    model_save_path = "whittle_model.pth"
    model = train_whittle_network(json_file, model_save_path=model_save_path)
