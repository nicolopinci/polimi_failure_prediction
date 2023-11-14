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
pen_values = ["l2"]

# Iterate through all combinations of parameter values
for pred_hours, read_minutes, C, pen in product(pred_hours_values, read_minutes_values, C_values, pen_values):
    command = f"python LogisticRegression.py --pred_hours {pred_hours} --read_minutes {read_minutes} --C {C} --penalty {pen}"
    
    # Run the command
    subprocess.run(command, shell=True)


pen_values = ["l1"]

# Iterate through all combinations of parameter values
for pred_hours, read_minutes, C, pen in product(pred_hours_values, read_minutes_values, C_values, pen_values):
    command = f"python LR_liblinear.py --pred_hours {pred_hours} --read_minutes {read_minutes} --C {C} --penalty {pen}"
    
    # Run the command
    subprocess.run(command, shell=True)