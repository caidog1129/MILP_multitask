import os
nthreads = 1
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)
import pickle
import argparse

from pyscipopt import Model
import pyscipopt as scip
import pyscipopt
import time

from MIPDataset import BipartiteNodeData, compute_mip_representation
from GAT import GATPolicy
import torch
import numpy as np
import pandas as pd
from ast import literal_eval

class MyEvent(pyscipopt.Eventhdlr):
    def eventinit(self):
        print("init event")
        self._start_time = time.monotonic()
        self.scip_log = [[],[]]
        self.start_time = time.monotonic()
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self):
        print("exit event")

    def eventexec(self, event):
        print("exec event")
        self.end_time = time.monotonic()
        sol = self.model.getBestSol()
        obj = self.model.getSolObjVal(sol)
        self.scip_log[0].append(obj)
        self.scip_log[1].append(self.end_time - self.start_time)
        self.start_time = self.end_time

def solve_milp(params=None, instance=""):
    model = Model()
    model.readProblem(instance)
    model = model.__repr__.__self__
    event = MyEvent()
    model.includeEventhdlr(
        event,
        "",
        ""
    )
    model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
    model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    model.disablePropagation()
    model.setParam("limits/memory", 12*1024)
    model.setParam("limits/time", 15 * 60)
    if params:
        params = {k: params[k] for k in params}
        print(params)
        model.setParams(params)
    model.hideOutput()

    model.optimize()

    # solution
    sol = model.getBestSol()
    primal = model.getPrimalbound()
    dual = model.getDualbound()
    time = model.getSolvingTime()

    return event

# Define the categorical mappings for one-hot encoding
categorical_params = {
    'branching/scorefunc': ['s', 'p', 'q'],
    'branching/lpgainnormalize': ['d', 'l', 's'],
    'lp/pricing': ['l', 'a', 'f', 'p', 's', 'q', 'd'],
    'nodeselection/childsel': ['d', 'u', 'p', 'i', 'l', 'r', 'h']
}

# Define the max values for normalization
max_values = {
    'branching/clamp': 0.5,
    'branching/midpull': 1,
    'branching/midpullreldomtrig': 1,
    'branching/scorefac': 1,
    'cutselection/hybrid/minortho': 1,
    'cutselection/hybrid/minorthoroot': 1,
    'lp/colagelimit': 2147483647,
    'lp/rowagelimit': 2147483647,
    'separating/cutagelimit': 2147483647,
    'separating/maxcutsgenfactor': 2147483647,
    'separating/maxcutsrootgenfactor': 2147483647,
    'separating/poolfreq': 65534
}

param_sequence = ['branching/clamp', 'branching/lpgainnormalize', 'branching/midpull',
    'branching/midpullreldomtrig',
    'branching/preferbinary',
    'branching/scorefac',
    'branching/scorefunc',
    'cutselection/hybrid/minortho',
    'cutselection/hybrid/minorthoroot',
    'lp/colagelimit',
    'lp/pricing',
    'lp/rowagelimit',
    'nodeselection/childsel',
    'separating/cutagelimit',
    'separating/maxcutsgenfactor',
    'separating/maxcutsrootgenfactor',
    'separating/poolfreq']

# Function to reverse one-hot encoding by choosing the index of the largest value
def reverse_one_hot(encoded_list, category_list):
    index = np.argmax(encoded_list)
    return category_list[index]

# Function to convert back normalized values
def reverse_normalize(value, max_value):
    return value * max_value

# Main function to decode parameters
def decode_params(encoded_values):
    result = {}
    idx = 0  # Index to track position in the encoded_values list

    for key in param_sequence:
        if key in categorical_params:
            # Length of the one-hot encoded list for this parameter
            length = len(categorical_params[key])
            one_hot_encoded = encoded_values[idx:idx + length]
            decoded_value = reverse_one_hot(one_hot_encoded, categorical_params[key])
            result[key] = decoded_value
            idx += length  # Move index forward by the length of the one-hot encoding
        elif key in max_values:
            # Get the next value and reverse normalization
            normalized_value = encoded_values[idx]
            if normalized_value < 0:
                normalized_value = 0
            if normalized_value > 1:
                normalized_value = 1
            original_value = reverse_normalize(normalized_value, max_values[key])
            result[key] = original_value
            idx += 1  # Move index forward by 1
        else:
            # If it's a boolean, decode directly
            if encoded_values[idx] <= 0.5:
                result[key] = 0
            else:
                result[key] = 1
            idx += 1  # Move index forward by 1
    
    return result

def evaluate(name, expname, instance):
    os.makedirs('results/' + expname + '/' + name + '/', exist_ok=True)
    
    data = compute_mip_representation(instance)
    
    saved_dict = torch.load("pretrain/" + expname.split("_")[0] + "_" + name + "/model_best.pth", map_location=torch.device('cpu'))
    model = GATPolicy()
    model.load_state_dict(saved_dict)
    model.eval()

    data.x_vars_batch = torch.zeros(data.x_vars.shape[0]).long()
    data.x_cons_batch = torch.zeros(data.x_cons.shape[0]).long()

    out = model(data.x_cons,
          data.edge_index_cons_to_vars,
          data.edge_attr,
          data.x_vars,
          data.x_vars_batch,
          data.x_cons_batch).detach()[0].numpy()

    params = decode_params(out)
    
    event = solve_milp(params = params, instance = instance)

    with open('results/' + expname + '/' + name + '/' + instance.split("/")[-1], "wb") as fp:
        pickle.dump(event.scip_log, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--expname", required=True)
    parser.add_argument("--models", nargs='+', required=True, help="List of models to evaluate")
    
    args = parser.parse_args()

    # Ensure directory structure exists
    # os.makedirs(f'results/{args.expname}/smac/', exist_ok=True)

    # df = pd.read_csv("datasets/" + args.expname + "_test/" + args.instance.split("/")[-1],index_col=0)
    # df = df.sort_values("0")

    # # Solve MILP and store logs
    # event = solve_milp(instance=args.instance, params = literal_eval(df.iloc[0]["1"]))
    # log_filename = f'results/{args.expname}/smac/{os.path.basename(args.instance)}'
    
    # with open(log_filename, "wb") as fp:
    #     pickle.dump(event.scip_log, fp)
    
    # # Loop through evaluations
    for model in args.models:
        evaluate(model, args.expname, args.instance)

    # Ensure directory structure exists
    os.makedirs(f'results/{args.expname}/scip/', exist_ok=True)

    # Solve MILP and store logs
    event = solve_milp(instance=args.instance)
    log_filename = f'results/{args.expname}/scip/{os.path.basename(args.instance)}'
    
    with open(log_filename, "wb") as fp:
        pickle.dump(event.scip_log, fp)