import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from supervised import Actor, ResidualActor,TransformerActor,PolyActor

# Load the trained model
model_path = "whittle_model.pth"
model = Actor(state_dim=1)
#model=ResidualActor(state_dim=1)
#model=TransformerActor(state_dim=1)
#model=PolyActor(state_dim=1)
model.load_state_dict(torch.load(model_path))
model.eval()

p_lists=np.linspace(start=0.2, stop=0.8, num=10)
# Generate states from 1 to 100
states = np.arange(1, 101, dtype=np.float32).reshape(-1, 1)

# Plot results
plt.figure(figsize=(10, 6))

for p in p_lists:
    q=p
    inputs = np.hstack((states, np.full_like(states, p), np.full_like(states, q), np.full_like(states,99)))
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
