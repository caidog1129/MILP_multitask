import pandas as pd
import gurobipy as grb
from ast import literal_eval
import argparse
import os

def main(instance_dir, exp_name):
    env = grb.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()

    df = pd.read_csv("datasets/" + exp_name + "/BD/" + instance_dir.split("/")[-1], sep=";")

    m = grb.read(instance_dir, env=env)
    m.optimize()
    baseline = m.Runtime
    
    previous_backdoor_lists = []  # To store backdoor lists from previous runs
    optimization_count = 0  # Counter for the number of optimizations
    
    for index, row in df.sort_values("reward", ascending=False).iterrows():
        if optimization_count >= 50:
            break  # Stop once 50 unique optimizations are done
        
        current_backdoor_list = literal_eval(row["backdoor_list"])  # Parse the backdoor list

        # Check if the reward or backdoor list is a duplicate
        is_duplicate_backdoor = any(set(current_backdoor_list) == set(prev_list) for prev_list in previous_backdoor_lists)

        if is_duplicate_backdoor:
            print(f"Skipping row {index} as the backdoor_list or reward is identical to a previously used one.")
            continue  # Skip if this backdoor list or reward is fully identical to a previously used one

        previous_backdoor_lists.append(current_backdoor_list)  # Add the new backdoor list to the record

        # Load the instance and set priorities
        m = grb.read(instance_dir, env=env)
        for i in range(len(m.getVars())):
            if i in current_backdoor_list:
                m.getVars()[i].BranchPriority = 2
            else:
                m.getVars()[i].BranchPriority = 1
        
        m.update()
        m.optimize()

        df.loc[index, "run_time"] = m.Runtime / baseline
        optimization_count += 1  # Increment the optimization counter

    df = df.dropna()
    df = df.sort_values("run_time")
    
    os.makedirs("datasets/" + exp_name + "/SOL/", exist_ok=True)
    df.to_csv("datasets/" + exp_name + "/SOL/" + instance_dir.split("/")[-1])

if __name__ == '__main__':
    parser_main = argparse.ArgumentParser()

    parser_main.add_argument("--instance_dir", type=str)
    parser_main.add_argument("--exp_name", type=str)
    args_main = parser_main.parse_args()

    main(instance_dir=args_main.instance_dir, exp_name=args_main.exp_name)
