import numpy as np
import json
from tqdm import tqdm

def Ib(a, d, lambd):
    x = (d + 0.5 * lambd * a * (a - 1)) / (1 - lambd + a * lambd)
    if d > (lambd / 2) * a**2 + (1 - lambd / 2) * a:
        return 0.5 * x**2 + (1 / lambd - 0.5) * x
    else:
        return d / lambd

def whittle_index(a_max, d_max, lambd):
    whittle_indices = np.zeros((a_max + 1, d_max + 1))
    
    with tqdm(total=a_max * d_max, desc="Computing Whittle Index", unit="points") as pbar:
        for a in range(1, a_max + 1):
            for d in range(d_max + 1):
                whittle_indices[a, d] = Ib(a, d, lambd)
                pbar.update(1)
    
    whittle_indices_dict = {str((a, d)): whittle_indices[a, d] for a in range(a_max + 1) for d in range(d_max + 1)}
    with open("closed_form_whittle_indices.json", "w") as f:
        json.dump(whittle_indices_dict, f)
    
    max_value = np.max(whittle_indices)
    min_value = np.min(whittle_indices)
    print(f"Max Whittle Index: {max_value}")
    print(f"Min Whittle Index: {min_value}")
    
    return whittle_indices

# 示例调用
whittle_indices = whittle_index(10, 10, 0.5)
print("Whittle Index Table Saved as JSON.")
