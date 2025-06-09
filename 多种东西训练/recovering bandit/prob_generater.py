import json
import numpy as np
import random

def generate_uniform_shuffled_theta_pairs(
    k_max=100,
    theta0_range=(10.0, 5.0),
    theta1_range=(0.2, 1.0),
    theta2=0.0,
    seed=42
):
    """
    在 theta0 × theta1 空间中生成均匀网格，并打乱顺序，使得前 k 个点也近似均匀
    """
    grid_size = int(np.ceil(np.sqrt(k_max)))
    theta0_vals = np.linspace(theta0_range[0], theta0_range[1], grid_size)
    theta1_vals = np.linspace(theta1_range[0], theta1_range[1], grid_size)

    # 生成网格组合
    theta_pairs_all = [(float(t0), float(t1), theta2) for t0 in theta0_vals for t1 in theta1_vals]

    # 打乱顺序
    random.seed(seed)
    random.shuffle(theta_pairs_all)

    # 截断前 k_max 个
    return theta_pairs_all[:k_max]

if __name__ == '__main__':
    max_state = 100
    k_max = 100
    theta_pairs = generate_uniform_shuffled_theta_pairs(k_max)

    output = {
        "maxWait": max_state,
        "theta_pairs": theta_pairs
    }

    with open("theta_pairs_recovering.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"✔ 已生成 {k_max} 个均匀打散的 theta 参数组，保存至 theta_pairs_recovering.json")
