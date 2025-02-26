import json
import copy
import numpy as np
import os
from singlearm import SingleArmAoIEnv

def load_json(filename):
    with open(filename, 'r') as file:
        data=json.load(file)
    return data

def simulate(envs, whittle_table, steps=30, runs=10):
    total_discounted_reward = 0
    for run in range(runs):
        envs_copy = copy.deepcopy(envs)
        discounted_reward = 0
        for t in range(steps):
            whittle_values = [(whittle_table[f'arm_{i}'][env.state[0]][env.state[1]], i) for i, env in enumerate(envs_copy)]
            whittle_values.sort(reverse=True, key=lambda x: x[0])
            selected_arms = [x[1] for x in whittle_values[:5]]
            reward_t = sum(envs_copy[i].step(1 if i in selected_arms else 0)[1] for i in range(len(envs_copy)))
            discounted_reward += reward_t * (0.99 ** t)
        total_discounted_reward += discounted_reward
    return total_discounted_reward / runs

initial_states = load_json('integer_pairs.json')
arm_prob = load_json('probability_pairs_M5_N10.json')

discounts = np.arange(0.05, 1.0, 0.05)
output_folder = "simulation_results"
os.makedirs(output_folder, exist_ok=True)

for discount in discounts:
    whittle_filename = f"whittle_results/whittle_M5_N10_discount{discount:.2f}.json"
    if not os.path.exists(whittle_filename):
        print(f"Skipping discount={discount:.2f}, Whittle index file not found.")
        continue
    
    whittle_index = load_json(whittle_filename)
    results = []
    for i in range(len(initial_states)):
        environments = [SingleArmAoIEnv(pg=arm_prob[j]['pg'], ps=arm_prob[j]['ps']) for j in range(len(initial_states[i]))]
        for env, state in zip(environments, initial_states[i]):
            env.reset()
            env.state = state
        
        simulated_reward = simulate(copy.deepcopy(environments), whittle_index, steps=30)
        results.append({"initial_state_index": i, "discount": discount, "reward": simulated_reward})
    
    output_file = os.path.join(output_folder, f"simulation_results_discount{discount:.2f}.json")
    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)
    print(f"Simulation for discount={discount:.2f} complete. Results saved to '{output_file}'.")
