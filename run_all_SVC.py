import argparse
import subprocess
from itertools import product

# Parse command-line arguments
parser = argparse.ArgumentParser()
args = parser.parse_args()

# Define the values for pred_hours, read_minutes, and C
pred_hours_values = [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
read_minutes_values = [10, 15, 20, 25, 30, 35]
C_values = [0.5, 1, 1.5]

# Iterate through all combinations of parameter values
for pred_hours, read_minutes, C in product(pred_hours_values, read_minutes_values, C_values):
    command = f"python SVC.py --pred_hours {pred_hours} --read_minutes {read_minutes} --C {C}"
    
    # Run the command
    subprocess.run(command, shell=True)