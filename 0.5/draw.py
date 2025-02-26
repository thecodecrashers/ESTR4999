import json
import matplotlib.pyplot as plt

# 读取JSON文件
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# 提取奖励数据
def extract_rewards(data):
    initial_states = [entry['initial_state_index'] for entry in data]
    baseline_rewards = [entry['baseline_reward'] for entry in data]
    modified_rewards = [entry['modified_reward'] for entry in data]
    wiql_rewards = [entry['WIQL_reward'] for entry in data]
    return initial_states, baseline_rewards, modified_rewards, wiql_rewards

# 绘制奖励比较图
def plot_rewards(initial_states, baseline_rewards, modified_rewards, wiql_rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(initial_states, baseline_rewards, label='Baseline Rewards', marker='o')
    plt.plot(initial_states, modified_rewards, label='Modified Rewards', marker='x')
    plt.plot(initial_states, wiql_rewards, label='WIQL Rewards', marker='s')

    plt.title('Reward Comparison Across Initial States')
    plt.xlabel('Initial State Index')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "simulation_results.json"  # 替换为您的文件路径
    data = load_data(file_path)

    initial_states, baseline_rewards, modified_rewards, wiql_rewards = extract_rewards(data)
    plot_rewards(initial_states, baseline_rewards, modified_rewards, wiql_rewards)
