import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

class LineEnv:
    """
    Simple line environment:

    - State ranges from 0 up to OptX (e.g., 99).
    - If we take action=1 on this arm:
        reward = 1
        with probability p => state resets to 0
        else => state = min(state + 1, OptX)
    - If we take action=0 (do nothing):
        reward = 0
        with probability q => state = min(state + 1, OptX)
        else => state = 0
    """
    def __init__(self, p, q, OptX=99, seed=42):
        self.p = p
        self.q = q
        self.OptX = OptX
        self.rng = random.Random(seed)
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        """
        Returns: (next_state, reward, done, info)
        We do not use 'done' or 'info' here, so set them to False, {}.
        """
        reward = 0
        if action == 1:
            # Taking action => reward=1
            reward = 1
            # With probability p => reset to 0, else increment
            if self.rng.random() < self.p:
                self.state = 0
            else:
                self.state = min(self.state + 1, self.OptX)
        else:
            # Doing nothing => reward=0
            # With probability q => state increments, else resets
            if self.rng.random() < self.q:
                self.state = min(self.state + 1, self.OptX)
            else:
                self.state = 0

        return self.state, reward, False, {}


def main():
    # ---------------------- 1) Load Whittle indices -------------------------
    with open("whittle_indices.json", "r") as f:
        whittle_data = json.load(f)

    # We'll organize whittle values in a dict keyed by (p, q, state) => whittle_val
    # (Because the JSON has multiple entries, you need to parse them into a lookup table.)
    whittle_dict = {}
    for entry in whittle_data:
        p_val = entry["p"]
        q_val = entry["q"]
        state_val = entry["state"]
        w_val = entry["whittle"]
        # Store in a dictionary using (p_val, q_val, state_val) as key
        whittle_dict[(p_val, q_val, state_val)] = w_val

    # ---------------------- 2) Create 10 arms with lineEnv ------------------
    # Let p, q go from 0.2 to 0.8 in uniform steps (9 intervals => i in [0..9]).
    # This replicates the approach in your original script.
    num_arms = 10
    p_vals = np.linspace(0.2, 0.8, num_arms)
    arms = []
    for i in range(num_arms):
        p_val = p_vals[i]
        q_val = p_val  # from your original approach, we set q = p
        env = LineEnv(p=p_val, q=q_val, OptX=99, seed=42 + i)
        arms.append(env)

    # Simulation config
    total_steps = 10000
    warmup_steps = 2000
    report_interval = 200
    top_k = 3

    total_reward_after_warmup = 0.0
    steps_after_warmup = 0

    # For plotting average reward across time
    report_records = []

    # ---------------------- 3) Run the simulation loop ----------------------
    for step_i in range(1, total_steps + 1):
        # 3.1) For each arm, get Whittle index from whittle_dict
        # clamp state to [0, 99] (just in case)
        whittle_values = []
        for arm_idx, env in enumerate(arms):
            st = max(0, min(env.state, 99))
            # Use (p, q, st) to look up whittle
            w_val = whittle_dict[(env.p, env.q, st)]
            whittle_values.append((w_val, arm_idx))

        # 3.2) Sort arms by whittle value, pick top_k
        whittle_values.sort(key=lambda x: x[0], reverse=True)
        chosen_indices = [pair[1] for pair in whittle_values[:top_k]]

        # 3.3) Perform actions and get total reward from this step
        step_reward_sum = 0.0
        for arm_idx, env in enumerate(arms):
            if arm_idx in chosen_indices:
                # action=1
                _, reward, _, _ = env.step(action=1)
            else:
                # action=0
                _, reward, _, _ = env.step(action=0)
            step_reward_sum += reward

        # 3.4) Accumulate reward only after warmup
        if step_i > warmup_steps:
            total_reward_after_warmup += step_reward_sum
            steps_after_warmup += 1

            # Possibly record intermediate results for plotting
            if step_i % report_interval == 0:
                avg_r = total_reward_after_warmup / steps_after_warmup
                report_records.append((step_i, avg_r))

    # ---------------------- 4) Plot and save the result ---------------------
    if report_records:
        steps_plt, avgR_plt = zip(*report_records)
        plt.figure(figsize=(7, 5))
        plt.plot(steps_plt, avgR_plt, marker='o', label=f"Top-{top_k} Whittle selection avg reward")
        plt.xlabel("Global Step")
        plt.ylabel("Average Reward")
        plt.title("Baseline (Whittle Index) Multi-Arms (Top-k = {})".format(top_k))
        plt.legend()
        out_img = "baseline_long.png"
        plt.savefig(out_img)
        plt.show()
        print(f"[Info] Final reported average reward = {avgR_plt[-1]}")
        print(f"Plot saved to {out_img}")
    else:
        print("[Info] No reported data (possibly your warmup_steps >= total_steps).")


if __name__ == "__main__":
    main()
