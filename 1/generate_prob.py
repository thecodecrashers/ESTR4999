import json
import numpy as np
import os

def generate_probability_file(N, M, seed=42, file_prefix="probability_pairs"):
    """
    Generate a JSON file containing N probability pairs for a multi-armed bandit problem.

    Args:
        N (int): Total number of arms.
        M (int): Number of arms that can be used simultaneously.
        seed (int): Random seed for reproducibility.
        file_prefix (str): Prefix for the output file name.

    Returns:
        str: The name of the generated JSON file.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate N probability pairs (pg, ps) where pg and ps are in [0, 1]
    probability_pairs = [{"pg": round(np.random.rand(), 2), "ps": round(np.random.rand(), 2)} for _ in range(N)]

    # Construct file name
    file_name = f"{file_prefix}_M{M}_N{N}.json"

    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Ensure the file is saved in the script's directory
    file_path = os.path.join(script_dir, file_name)

    # Write probability pairs to a JSON file
    with open(file_path, "w") as file:
        json.dump(probability_pairs, file, indent=4)

    print(f"Probability pairs file generated: {file_path}")
    return file_path

# Example usage
if __name__ == "__main__":
    # Parameters
    N = 10  # Total number of arms
    M = 5   # Number of arms that can be used simultaneously
    seed = 42  # Random seed

    # Generate the JSON file
    generate_probability_file(N, M, seed)
