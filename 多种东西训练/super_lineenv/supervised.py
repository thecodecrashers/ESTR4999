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

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0.01)
            nn.init.constant_(m.bias, 0.01)

    def forward(self, state_env):
        x = self.fc1(state_env)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        logit = self.fc5(x)  
        return logit

# Training the model
def train_whittle_network(json_file, num_epochs=5000, batch_size=64, learning_rate=0.001, model_save_path="whittle_model.pth"):
    dataset = WhittleDataset(json_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = Actor(state_dim=1)  # state + p, q, optx
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
