import argparse
import subprocess
from itertools import product

# Parse command-line arguments
parser = argparse.ArgumentParser()
args = parser.parse_args()

# Define the values for pred_hours, read_minutes, and C
pred_hours_values = [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
read_minutes_values = [10, 15, 20, 25, 30, 35]

# Iterate through all combinations of parameter values
for pred_hours, read_minutes in product(pred_hours_values, read_minutes_values):
    if (pred_hours == 0.25 and read_minutes == 35) or (pred_hours == 0.5 and read_minutes == 35) or (pred_hours == 0.25 and read_minutes == 30):
        continue
    
    command = f"python LSTM_archBCEUnidirectional.py --pred_hours {pred_hours} --read_minutes {read_minutes}"
    
    # Run the command
    subprocess.run(command, shell=True)