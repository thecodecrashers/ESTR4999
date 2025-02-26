import json
import numpy as np
import os
from singlearm import SingleArmAoIEnv
from tqdm import tqdm

def calculate_q_passive(env, tolerance=0.01, max_iterations=1000):
    """
    Calculate the converged passive action-value function Q(X_i, 0).

    Args:
        env: The SingleArmAoIEnv environment.
        tolerance: Convergence tolerance.
        max_iterations: Maximum number of iterations.

    Returns:
        q_passive_table: Converged Q(X_i, 0) table.
    """
    state, discount, max_a, max_d, pg, ps, _ = env.reset()
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

    for iteration in tqdm(range(max_iterations), desc="Converging Q_passive"):
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

def calculate_lambda(env, q_passive_table):
    """
    Calculate the state-dependent Whittle index \hat{\lambda}(X_i).

    Args:
        env: The SingleArmAoIEnv environment.
        q_passive_table: Converged Q(X_i, 0) table.

    Returns:
        lambda_table: The Whittle index table.
    """
    state, discount, max_a, max_d, pg, ps, _ = env.reset()
    lambda_table = np.zeros((max_a + 1, max_d + 1), dtype=np.float64)

    def reward(state, action):
        a, d = state
        return - (a + d)

    def cal_transitions(state, action):
        a, d = state
        transitions = []

        if action == 0:  # Passive action
            transitions.append(((min(a + 1, max_a), min(d + a, max_d)), pg))
            transitions.append(((min(a + 1, max_a), d), 1 - pg))
        else:  # Active action
            transitions.append(((1, 1), ps * pg))
            transitions.append(((1, min(d + a, max_d)), ps * (1 - pg)))
            transitions.append(((min(a + 1, max_a), min(d + a, max_d)), (1 - ps) * pg))
            transitions.append(((min(a + 1, max_a), d), (1 - ps) * (1 - pg)))

        return transitions

    for a in range(max_a + 1):
        for d in range(max_d + 1):
            state = (a, d)

            # Compute components of \hat{\lambda}(X_i)
            reward_diff = reward(state, 1) - reward(state, 0)
            future_diff = discount * (
                sum(p * q_passive_table[s] for s, p in cal_transitions(state, 1)) -
                sum(p * q_passive_table[s] for s, p in cal_transitions(state, 0))
            )

            lambda_table[a, d] = reward_diff + future_diff

    return lambda_table

def calculate_and_save_modified_whittle_indices(probability_file, M, N, output_prefix="modified_whittle"):
    """
    Calculate and save state-dependent Whittle indices for a multi-arm problem based on probabilities.

    Args:
        probability_file (str): JSON file containing probability pairs.
        M (int): Number of arms used simultaneously.
        N (int): Total number of arms.
        output_prefix (str): Prefix for the output file name.
    """
    # Load probability pairs
    with open(probability_file, "r") as file:
        probability_pairs = json.load(file)

    # Ensure probabilities are valid
    assert len(probability_pairs) == N, "Number of probability pairs does not match N."

    # Calculate Whittle indices for each arm
    whittle_indices = {}
    for i, probs in enumerate(tqdm(probability_pairs, desc="Calculating Whittle Indices")):
        pg, ps = probs["pg"], probs["ps"]
        env = SingleArmAoIEnv(pg=pg, ps=ps)

        # Step 1: Compute Q_passive
        q_passive_table = calculate_q_passive(env)

        # Step 2: Compute Lambda table
        lambda_table = calculate_lambda(env, q_passive_table)

        whittle_indices[f"arm_{i}"] = lambda_table.tolist()

    # Construct output file name
    output_file = f"{output_prefix}_M{M}_N{N}.json"

    # Save Whittle indices to file
    with open(output_file, "w") as file:
        json.dump(whittle_indices, file, indent=4)

    print(f"Modified Whittle indices saved to: {output_file}")

# Example usage
if __name__ == "__main__":
    probability_file = "probability_pairs_M5_N10.json"  # Example input file
    M = 5
    N = 10
    calculate_and_save_modified_whittle_indices(probability_file, M, N)
