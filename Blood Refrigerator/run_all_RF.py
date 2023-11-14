import subprocess

def execute_logistic_regression(folder, RW_minutes):
    command = ["python", "RF.py", str(folder), str(RW_minutes)]
    subprocess.run(command, stdout=subprocess.PIPE, text=True)

if __name__ == "__main__":
    pws = [0.5, 1.0, 1.5, 2.0]
    RW_minutes_list = [10, 15, 20, 25, 30]
    
    for pw in pws:
        for RW_minutes in RW_minutes_list:
            print("PW:", pw, "RW:", RW_minutes)
            execute_logistic_regression(pw, RW_minutes)
