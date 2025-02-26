import json
import copy
from singlearm import SingleArmAoIEnv
def load_json(filename):
    with open(filename, 'r') as file:
        data=json.load(file)
    return data

def simulate(envs, whittle_table, steps=30000, runs=10):
    total_discounted_reward = 0
    for run in range(runs):
        # Reset environments for each run
        envs_copy = copy.deepcopy(envs)
        discounted_reward = 0
        for t in range(steps):
            whittle_values = []
            for i, env in enumerate(envs_copy):
                a, d = env.state
                whittle_values.append((whittle_table[f'arm_{i}'][a][d], i))
            whittle_values.sort(reverse=True, key=lambda x: x[0])
            selected_arms = [x[1] for x in whittle_values[:5]]

            # Perform actions on selected arms
            reward_t = 0
            for i, env in enumerate(envs_copy):
                action = 1 if i in selected_arms else 0  # 1: Transmit, 0: No transmit
                _, reward, _, _, _ = env.step(action)
                reward_t += reward

            discounted_reward += reward_t * (0.99 ** t)

        total_discounted_reward += discounted_reward

    # Return the average discounted reward over all runs
    return total_discounted_reward / runs


initial_states=load_json('integer_pairs.json')
#print(initial_states[0])
arm_prob=load_json('probability_pairs_M5_N10.json')

baseline_whittles=load_json('baseline_whittle_M5_N10.json')

modified_whittles=load_json('modified_whittle_M5_N10.json')

WIQL_whittle=load_json('WIQL_N10_M5.json')

results=[]

for i in range(len(initial_states)):
    environments=[]
    for j in range(len(initial_states[i])):
        initial_state=initial_states[i][j]
        pg,ps=arm_prob[j]['pg'],arm_prob[j]['ps']
        env=SingleArmAoIEnv(pg=pg,ps=ps)
        env.reset()
        env.state=initial_state
        environments.append(env)
    env_baseline=copy.deepcopy(environments)
    env_modified=copy.deepcopy(environments)
    env_WIQL=copy.deepcopy(environments)
    baseline_reward = simulate(env_baseline, baseline_whittles, steps=30)
    modified_reward = simulate(env_modified, modified_whittles, steps=30)
    WIQL_reward = simulate(env_WIQL, WIQL_whittle, steps=30)

    # Store results
    results.append({
        "initial_state_index": i,
        "baseline_reward": baseline_reward,
        "modified_reward": modified_reward,
        "WIQL_reward": WIQL_reward
    })

with open('simulation_results.json', 'w') as outfile:
    json.dump(results, outfile, indent=4)

print("Simulation complete. Results saved to 'simulation_results.json'.")