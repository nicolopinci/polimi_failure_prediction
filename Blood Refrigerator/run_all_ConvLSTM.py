import subprocess
import os
def execute_transformer(folder, RW_minutes):
    command = ["python", "ConvLSTM.py", str(folder), str(RW_minutes)]
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)

if __name__ == "__main__":
    pws = [0.5, 1.0, 1.5, 2.0]
    RW_minutes_list = [10, 15, 20, 25, 30]
    for pw in pws:
        for RW_minutes in RW_minutes_list:
            print("PW:", pw, "RW:", RW_minutes)
            execute_transformer(pw, RW_minutes)
