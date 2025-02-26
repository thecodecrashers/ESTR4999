import json
import numpy as np
import os
from singlearm import SingleArmAoIEnv
from tqdm import tqdm

def calculate_whittle_index(env, tolerance=0.01, max_iterations=1000, outer_iterations=30):
    state, discount, max_a, max_d, pg, ps, _ = env.reset()

    # Initialize value and Whittle tables as numpy arrays
    value_table = np.zeros((max_a + 1, max_d + 1), dtype=np.float64)
    whittle_table = np.zeros((max_a + 1, max_d + 1), dtype=np.float64)
    min_whittle_table = np.full((max_a + 1, max_d + 1), -1000, dtype=np.float64)
    max_whittle_table = np.full((max_a + 1, max_d + 1), 1000, dtype=np.float64)

    # Reward function
    def reward(state):
        a, d = state
        return - (a + d)

    # Transition calculation function
    transition_cache = {}
    def cal_transitions(state, action):
        if (state, action) in transition_cache:
            return transition_cache[(state, action)]
        a, d = state
        transitions = []
        
        if action == 0:  # No transmission
            transitions.append(((min(a + 1, max_a), min(d + a, max_d)), pg))
            transitions.append(((min(a + 1, max_a), d), 1 - pg))
        else:  # Attempt transmission
            transitions.append(((1, 1), ps * pg))
            transitions.append(((1, min(d + a, max_d)), ps * (1 - pg)))
            transitions.append(((min(a + 1, max_a), min(d + a, max_d)), (1 - ps) * pg))
            transitions.append(((min(a + 1, max_a), d), (1 - ps) * (1 - pg)))
        transition_cache[(state, action)] = transitions
        return transitions

    # Value iteration function
    def run_value_iteration(cost):
        nonlocal value_table
        for iteration in range(max_iterations):
            new_value_table = value_table.copy()
            delta = 0

            for a in range(max_a + 1):
                for d in range(max_d + 1):
                    state = (a, d)
                    
                    # Calculate passive and active values
                    passive_value = sum(p * (reward(s) + discount * value_table[s]) for s, p in cal_transitions(state, 0))
                    active_value = sum(p * (reward(s) + discount * value_table[s]) for s, p in cal_transitions(state, 1)) - cost
                    
                    # Select the best value
                    best_value = max(passive_value, active_value)
                    new_value_table[a, d] = best_value
                    delta = max(delta, abs(value_table[a, d] - best_value))

            # Update global value table
            value_table = new_value_table

            # Check for convergence
            if delta < tolerance:
                break

        return value_table

    # Outer loop for Whittle index calculation
    for a in range(max_a + 1):
        if a==0:
            continue
        for d in range(max_d + 1):
            if d==0:
                continue
            state = (a, d)
            whittle_table[state] = 0.5 * (max_whittle_table[state] + min_whittle_table[state])
            while ((max_whittle_table[state] - min_whittle_table[state] )>0.1):
                run_value_iteration(whittle_table[state])
                
                # Calculate passive and active values
                passive_value = sum(p * (reward(s) + discount * value_table[s]) for s, p in cal_transitions(state, 0))
                active_value = sum(p * (reward(s) + discount * value_table[s]) for s, p in cal_transitions(state, 1)) - whittle_table[state]
                
                # Update Whittle bounds
                if active_value > passive_value:
                    min_whittle_table[state] = whittle_table[state]
                elif active_value < passive_value:
                    max_whittle_table[state] = whittle_table[state]
                else:
                    break
                whittle_table[state] = 0.5 * (max_whittle_table[state] + min_whittle_table[state])

    return whittle_table


def calculate_and_save_baseline_whittle_indices(probability_file, M, N, output_prefix="baseline_whittle"):
    """
    Calculate and save Whittle indices for a multi-arm problem based on probabilities.

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
    with tqdm(total=N, desc="Calculating Whittle Indices") as pbar:
        for i, probs in enumerate(probability_pairs):
            pg, ps = probs["pg"], probs["ps"]
            env = SingleArmAoIEnv(pg=pg, ps=ps)
            whittle_table = calculate_whittle_index(env)
            whittle_indices[f"arm_{i}"] = whittle_table.tolist()
            pbar.update(1)

    # Construct output file name
    output_file = f"{output_prefix}_M{M}_N{N}.json"

    # Save Whittle indices to file
    with open(output_file, "w") as file:
        json.dump(whittle_indices, file, indent=4)

    print(f"Baseline Whittle indices saved to: {output_file}")

# Example usage
if __name__ == "__main__":
    probability_file = "probability_pairs_M5_N10.json"  # Example input file
    M = 5
    N = 10
    calculate_and_save_baseline_whittle_indices(probability_file, M, N)
