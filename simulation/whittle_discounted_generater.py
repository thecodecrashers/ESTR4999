import json
import numpy as np
import os
from singlearm import SingleArmAoIEnv
from tqdm import tqdm

def calculate_q_passive(env, discount, tolerance=0.01, max_iterations=1000):
    state, _, max_a, max_d, pg, ps, _ = env.reset()
    q_passive_table = np.zeros((max_a + 1, max_d + 1), dtype=np.float64)

    def reward(state):
        a, d = state
        return - (a + d)

    def cal_transitions(state, action):
        a, d = state
        transitions = []
        if action == 0:  # Passive action
            transitions.append(((min(a + 1, max_a), min(d + a, max_d)), pg))
            transitions.append(((min(a + 1, max_a), d), 1 - pg))
        return transitions

    for _ in tqdm(range(max_iterations), desc=f"Converging Q_passive (discount={discount:.2f})"):
        new_q_passive_table = q_passive_table.copy()
        delta = 0
        for a in range(max_a + 1):
            for d in range(max_d + 1):
                state = (a, d)
                q_passive = sum(p * (reward(state) + discount * q_passive_table[s])
                                for s, p in cal_transitions(state, 0))
                new_q_passive_table[a, d] = q_passive
                delta = max(delta, abs(q_passive_table[a, d] - q_passive))
        q_passive_table = new_q_passive_table
        if delta < tolerance:
            break
    return q_passive_table

def calculate_lambda(env, q_passive_table, discount):
    state, _, max_a, max_d, pg, ps, _ = env.reset()
    lambda_table = np.zeros((max_a + 1, max_d + 1), dtype=np.float64)

    def reward(state, action):
        a, d = state
        return - (a + d)

    def cal_transitions(state, action):
        a, d = state
        transitions = []
        if action == 0:
            transitions.append(((min(a + 1, max_a), min(d + a, max_d)), pg))
            transitions.append(((min(a + 1, max_a), d), 1 - pg))
        else:
            transitions.append(((1, 1), ps * pg))
            transitions.append(((1, min(d + a, max_d)), ps * (1 - pg)))
            transitions.append(((min(a + 1, max_a), min(d + a, max_d)), (1 - ps) * pg))
            transitions.append(((min(a + 1, max_a), d), (1 - ps) * (1 - pg)))
        return transitions

    for a in range(max_a + 1):
        for d in range(max_d + 1):
            state = (a, d)
            reward_diff = reward(state, 1) - reward(state, 0)
            future_diff = discount * (
                sum(p * q_passive_table[s] for s, p in cal_transitions(state, 1)) -
                sum(p * q_passive_table[s] for s, p in cal_transitions(state, 0))
            )
            lambda_table[a, d] = reward_diff + future_diff
    return lambda_table

def calculate_and_save_whittle_indices(probability_file, M, N, output_folder="whittle_results"):
    os.makedirs(output_folder, exist_ok=True)
    with open(probability_file, "r") as file:
        probability_pairs = json.load(file)
    assert len(probability_pairs) == N, "Number of probability pairs does not match N."
    
    discounts = np.arange(0.05, 1.0, 0.05)
    for discount in discounts:
        whittle_indices = {}
        for i, probs in enumerate(tqdm(probability_pairs, desc=f"Calculating Whittle Indices (discount={discount:.2f})")):
            pg, ps = probs["pg"], probs["ps"]
            env = SingleArmAoIEnv(pg=pg, ps=ps)
            q_passive_table = calculate_q_passive(env, discount)
            lambda_table = calculate_lambda(env, q_passive_table, discount)
            whittle_indices[f"arm_{i}"] = lambda_table.tolist()
        
        output_file = os.path.join(output_folder, f"whittle_M{M}_N{N}_discount{discount:.2f}.json")
        with open(output_file, "w") as file:
            json.dump(whittle_indices, file, indent=4)
        print(f"Whittle indices saved to: {output_file}")

# Example usage
if __name__ == "__main__":
    probability_file = "probability_pairs_M5_N10.json"
    M = 5
    N = 10
    calculate_and_save_whittle_indices(probability_file, M, N)
