import argparse
import subprocess
from itertools import product

# Parse command-line arguments
parser = argparse.ArgumentParser()
args = parser.parse_args()

# Define the values for pred_hours, read_minutes, and C
pred_hours_values = [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
read_minutes_values = [10, 15, 20, 25, 30, 35]
n_trees_values = [100, 150, 200]
max_features_values = ["sqrt", "log2", 0.33, 1]

# Iterate through all combinations of parameter values
for pred_hours, read_minutes, ntrees, max_features in product(pred_hours_values, read_minutes_values, n_trees_values, max_features_values):
    command = f"python RandomForest.py --pred_hours {pred_hours} --read_minutes {read_minutes} --ntrees {ntrees} --maxfeatures {max_features}"
    
    # Run the command
    subprocess.run(command, shell=True)