import gurobipy as grb
import os
nthreads = 1
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)
import pandas as pd
import argparse

from pyscipopt import Model
import pyscipopt as scip
import pyscipopt
import time

from MIPDataset import BipartiteNodeData, compute_mip_representation
from GAT import GATPolicy, GATPolicy_multitask
import torch
import numpy as np

env = grb.Env(empty=True)
env.setParam("OutputFlag",0)
env.setParam("Threads",1)
env.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance")
    parser.add_argument("--expname")
    parser.add_argument("--model")
    parser.add_argument("--multitask_model")
    # parser.add_argument("--multitask_model_v3")
    args = parser.parse_args()

    os.makedirs('results/' + args.expname, exist_ok=True)

    # Gurobi
    m = grb.read(args.instance, env = env)
    m.optimize()
    baseline = m.Runtime

    # GAT
    m = grb.read(args.instance, env = env)
    
    data = compute_mip_representation(args.instance)

    saved_dict = torch.load(args.model, map_location=torch.device('cpu'))
    model = GATPolicy()
    model.load_state_dict(saved_dict)
    model.eval()

    out = model(data.x_cons,
          data.edge_index_cons_to_vars,
          data.edge_attr,
          data.x_vars)

    values, indices = torch.topk(out, 8)

    for i in range(len(m.getVars())):
        if i in indices:
            m.getVars()[i].BranchPriority = 2
        else:
            m.getVars()[i].BranchPriority = 1
    m.update()
    m.optimize()
    gat = m.Runtime

    # Multitask
    m = grb.read(args.instance, env = env)
    
    data = compute_mip_representation(args.instance)

    saved_dict = torch.load(args.multitask_model, map_location=torch.device('cpu'))
    model = GATPolicy()
    model.load_state_dict(saved_dict)
    model.eval()

    out = model(data.x_cons,
          data.edge_index_cons_to_vars,
          data.edge_attr,
          data.x_vars)

    values, indices = torch.topk(out, 8)

    for i in range(len(m.getVars())):
        if i in indices:
            m.getVars()[i].BranchPriority = 2
        else:
            m.getVars()[i].BranchPriority = 1
    m.update()
    m.optimize()
    multitask = m.Runtime

    # Multitask_v3
    # m = grb.read(args.instance, env = env)
    
    # data = compute_mip_representation(args.instance)

    # saved_dict = torch.load(args.multitask_model_v3, map_location=torch.device('cpu'))
    # model = GATPolicy()
    # model.load_state_dict(saved_dict)
    # model.eval()

    # out = model(data.x_cons,
    #       data.edge_index_cons_to_vars,
    #       data.edge_attr,
    #       data.x_vars)

    # values, indices = torch.topk(out, 8)

    # for i in range(len(m.getVars())):
    #     if i in indices:
    #         m.getVars()[i].BranchPriority = 2
    #     else:
    #         m.getVars()[i].BranchPriority = 1
    # m.update()
    # m.optimize()
    # multitask_v3 = m.Runtime

    df = pd.DataFrame(columns = ['gurobi', 'gat', 'multitask'])
    df = df._append({'gurobi': baseline, 'gat': gat, 'multitask': multitask}, ignore_index=True)
    df.to_csv('results/' + args.expname + "/" + args.instance.split("/")[-1], index=False)