import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_whittle_indices(binary_file, closed_form_file, a_max=9, d_max=9):
    with open(binary_file, "r") as f:
        binary_data = json.load(f)
    with open(closed_form_file, "r") as f:
        closed_form_data = json.load(f)
    
    binary_indices = np.zeros((a_max + 1, d_max + 1))
    closed_form_indices = np.zeros((a_max + 1, d_max + 1))
    
    ad_values = []
    for a in range(a_max + 1):
        for d in range(d_max + 1):
            binary_indices[a, d] = binary_data.get(str((a, d)), 0)
            closed_form_indices[a, d] = closed_form_data.get(str((a, d)), 0)
            ad_values.append((a, d))
    
    plt.figure(figsize=(10, 6))
    
    binary_flattened = binary_indices.flatten()
    closed_form_flattened = closed_form_indices.flatten()
    ad_labels = [f"({a},{d})" for a, d in ad_values]
    
    plt.plot(range(len(ad_labels)), binary_flattened, label="Binary Whittle Index", linestyle='dashed')
    plt.plot(range(len(ad_labels)), closed_form_flattened, label="Closed-Form Whittle Index")
    
    plt.xticks(range(len(ad_labels)), ad_labels, rotation=90, fontsize=8)
    plt.xlabel("(a, d) values")
    plt.ylabel("Whittle Index")
    plt.title("Whittle Index Comparison")
    plt.legend()
    plt.grid()
    plt.show()

# 示例调用
plot_whittle_indices("binary_whittle_indices.json", "closed_form_whittle_indices.json", a_max=9, d_max=9)
