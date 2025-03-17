import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Load the trained model
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

# Load the trained model
model_path = "whittle_model.pth"
model = Actor(state_dim=1)
model.load_state_dict(torch.load(model_path))
model.eval()

p_lists=np.linspace(start=0.2, stop=0.8, num=10)
# Generate states from 1 to 100
states = np.arange(1, 101, dtype=np.float32).reshape(-1, 1)

# Plot results
plt.figure(figsize=(10, 6))

for p in p_lists:
    q=p
    inputs = np.hstack((states, np.full_like(states, p), np.full_like(states, q), np.zeros_like(states)))
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    with torch.no_grad():
        whittle_indices = model(inputs_tensor).numpy().flatten()
    plt.plot(states, whittle_indices, label=f"p={p}, q={q}")

plt.xlabel("State")
plt.ylabel("Whittle Index")
plt.title("Whittle Index Predictions for Different (p, q) Pairs")
plt.legend()
plt.grid()
plt.show()
