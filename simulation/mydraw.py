import json
import numpy as np
import os
import matplotlib.pyplot as plt

def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def plot_results(output_folder):
    strategies = ["baseline", "modified", "WIQL"]
    colors = ["b", "g", "r"]
    plt.figure(figsize=(8, 6))
    
    for strategy, color in zip(strategies, colors):
        discounts = []
        all_rewards = []
        for filename in os.listdir(output_folder):
            if filename.startswith(f"simulation_results_{strategy}_discount") and filename.endswith(".json"):
                discount = float(filename.split("discount")[1].split(".json")[0])
                data = load_json(os.path.join(output_folder, filename))
                rewards = [entry["reward"] for entry in data]  # Collect all rewards per discount
                discounts.extend([discount] * len(rewards))  # Repeat discount for each reward
                all_rewards.extend(rewards)
        
        plt.scatter(discounts, all_rewards, color=color, label=strategy, alpha=0.6)
    
    plt.xlabel("Discount Factor")
    plt.ylabel("Discounted Reward")
    plt.title("Full Simulation Results for Different Whittle Index Strategies")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_folder, "full_simulation_results_plot.png"))
    plt.show()

output_folder = "simulation_results"
plot_results(output_folder)
