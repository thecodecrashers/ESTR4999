import random
import json

def generate_integer_pairs(m):
    """
    Generate m groups of 10 pairs of integers, with values between 1 and 10 (inclusive).

    Args:
        m (int): Number of groups.

    Returns:
        list: A list of m groups, each containing 10 integer pairs.
    """
    all_groups = []
    for _ in range(m):
        group = [(random.randint(1, 10), random.randint(1, 10)) for _ in range(10)]
        all_groups.append(group)
    return all_groups

def save_to_json(data, filename):
    """
    Save data to a JSON file.

    Args:
        data (any): The data to save.
        filename (str): The name of the file to save the data to.
    """
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage
m = 100  # Number of groups to generate
result = generate_integer_pairs(m)

# Save the result to a JSON file
save_to_json(result, 'integer_pairs.json')

# Display the result
for i, group in enumerate(result):
    print(f"Group {i + 1}:")
    for pair in group:
        print(pair)
    print()
