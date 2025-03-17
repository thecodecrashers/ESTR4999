import json
import numpy as np
from tqdm import tqdm
from lineEnv import lineEnv

def calculate_whittle_index_lineenv(env, tolerance=0.01, max_iterations=1000):
    max_state = env.N - 1
    opt_state = env.OptX
    value_table = np.zeros(env.N, dtype=np.float64)
    whittle_table = np.zeros(env.N, dtype=np.float64)
    min_whittle_table = np.full(env.N, -1000, dtype=np.float64)
    max_whittle_table = np.full(env.N, 1000, dtype=np.float64)

    def reward(state):
        x = state[0]
        return 1 - ((x - opt_state)**2 / max(opt_state, max_state - opt_state)**2)

    def run_value_iteration(env, cost, discount=0.99, max_iterations=1000, tolerance=0.001):
        value_table = np.zeros(env.N, dtype=np.float64)

        for iteration in range(max_iterations):
            new_value_table = value_table.copy()
            delta = 0

            for x in range(env.N):
                passive_value = (
                    reward([x]) +
                    discount * (
                        env.q * value_table[max(0, x - 1)] +
                        (1 - env.q) * value_table[x]
                    )
                )
                active_value = (
                    reward([x]) - cost +
                    discount * (
                        env.p * value_table[min(max_state, x + 1)] +
                        (1 - env.p) * value_table[x]
                    )
                )
                new_value_table[x] = max(passive_value, active_value)
                delta = max(delta, abs(new_value_table[x] - value_table[x]))

            value_table[:] = new_value_table
            if delta < tolerance:
                break

        return value_table

    for state in range(env.N):
        while max_whittle_table[state] - min_whittle_table[state] > tolerance:
            whittle_table[state] = 0.5 * (max_whittle_table[state] + min_whittle_table[state])
            value_table = run_value_iteration(env, whittle_table[state])

            passive_value = (
                reward([state]) +
                env.q * value_table[max(0, state - 1)] +
                (1 - env.q) * value_table[state]
            )
            active_value = (
                reward([state]) - whittle_table[state] +
                env.p * value_table[min(max_state, state + 1)] +
                (1 - env.p) * value_table[state]
            )

            if active_value > passive_value:
                min_whittle_table[state] = whittle_table[state]
            elif active_value < passive_value:
                max_whittle_table[state] = whittle_table[state]
            else:
                break

    return whittle_table

def calculate_and_save_whittle_indices(N, num_arms, optx_list, p_list, q_list, ran_seed, output_file):
    whittle_indices = []

    with tqdm(total=num_arms, desc="Calculating Whittle Indices") as pbar:
        for i in range(num_arms):
            env = lineEnv(seed=ran_seed, N=N, OptX=optx_list, p=p_list[i], q=q_list[i])
            whittle_table = calculate_whittle_index_lineenv(env)
            
            for state in range(N):
                whittle_indices.append({
                    "p": p_list[i],
                    "q": q_list[i],
                    "OptX": optx_list,
                    "state": state,
                    "whittle": whittle_table[state]
                })
            
            pbar.update(1)

    with open(output_file, "w") as file:
        json.dump(whittle_indices, file, indent=4)

    print(f"Whittle indices saved to {output_file}")

if __name__ == "__main__":
    N = 100
    num_arms = 50
    ran_seed = 42
    optx_list = 99
    prob_values = np.linspace(start=0.2, stop=0.8, num=num_arms)
    p_list = prob_values
    q_list = p_list
    output_file = "whittle_indices.json"

    calculate_and_save_whittle_indices(N, num_arms, optx_list, p_list, q_list, ran_seed, output_file)
