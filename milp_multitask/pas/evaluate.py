import gurobipy as grb
from gurobipy import GRB
import os
nthreads = 1
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)
import pandas as pd
import argparse

import time

from MIPDataset import BipartiteNodeData, compute_mip_representation
from GAT import GATPolicy, GATPolicy_multitask
import torch
import numpy as np
import pickle


grb.setParam('LogToConsole', 1)

def mycallback(model, where):

    if where == GRB.Callback.MIPSOL:

        # Access solution values using the custom attribute model._vars
        # sol = model.cbGetSolution(model._vars)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        runtime = time.monotonic() - model._starttime
        log_entry_name = model.Params.LogFile

        log_entry = dict()
        log_entry['primal_bound'] = obj
        log_entry['solving_time'] = runtime
        var_index_to_value = dict()
        # for i in range(len(model._vars)):
        #     v_name = model._vars[i].VarName
        #     v_value = sol[i]
        #     var_index_to_value[v_name] = v_value

        # log_entry['var_name_to_value'] = copy.deepcopy(var_index_to_value)
        gurobi_log[log_entry_name].append(log_entry)

        print("New solution", obj, "Runtime", runtime)

DEVICE = torch.device("cpu")
gurobi_log = dict()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance")
    parser.add_argument("--expname")
    parser.add_argument("--model")
    parser.add_argument("--multitask_model")
    # parser.add_argument("--multitask_model_v2")
    
    args = parser.parse_args()
    TaskName=args.expname
    def test_hyperparam(task):
        '''
        set the hyperparams
        k_0, k_1,delta
        '''
        if "CA" in task:
            return 1500, 0, 0
        elif task == "ISv3":
            return 250, 250, 10
        elif task == "IS_hv3":
            return 500, 500, 10
        elif "MVC" in task:
            # return 500, 100, 15
            return 250, 50, 10
        
    k_0,k_1,delta=test_hyperparam(TaskName)
    
    os.makedirs('results/' + args.expname + '/gurobi/', exist_ok=True)
        
    m = grb.read(args.instance)
    m.Params.TimeLimit = 1000
    m.Params.Threads = 1
    m.Params.MIPFocus = 1
    m.Params.LogFile = 'gurobi'
    gurobi_log['gurobi'] = []
    m._starttime = time.monotonic()
    m.optimize(mycallback)
    with open('results/' + args.expname + '/gurobi/' + args.instance.split("/")[-1], "wb") as fp:
        pickle.dump(gurobi_log[m.Params.LogFile], fp)
    
    os.makedirs('results/' + args.expname + '/gat/', exist_ok=True)
    
    saved_dict = torch.load(args.model, map_location=torch.device('cpu'))
    model = GATPolicy()
    model.load_state_dict(saved_dict)
    model.eval()
    
    #get bipartite graph as input
    A, v_map, v_nodes, c_nodes, b_vars=compute_mip_representation(args.instance)
    constraint_features = c_nodes.cpu()
    # constraint_features[np.isnan(constraint_features)] = 1 #remove nan value
    variable_features = v_nodes
    edge_indices = A._indices()
    edge_features = A._values().unsqueeze(1)
    edge_features=torch.ones(edge_features.shape)
    
    #prediction
    BD = model(
        constraint_features.to(DEVICE),
        edge_indices.to(DEVICE),
        edge_features.to(DEVICE),
        variable_features.to(DEVICE),
    ).sigmoid().cpu().squeeze()
    
    #align the variable name betweend the output and the solver
    all_varname=[]
    for name in v_map:
        all_varname.append(name)
    binary_name=[all_varname[i] for i in b_vars]
    scores=[]#get a list of (index, VariableName, Prob, -1, type)
    for i in range(len(v_map)):
        type="C"
        if all_varname[i] in binary_name:
            type='BINARY'
        scores.append([i, all_varname[i], BD[i].item(), -1, type])
    
    scores.sort(key=lambda x:x[2],reverse=True)
    
    scores=[x for x in scores if x[4]=='BINARY']#get binary
    
    fixer=0
    #fixing variable picked by confidence scores
    count1=0
    for i in range(len(scores)):
        if count1<k_1:
            scores[i][3] = 1
            count1+=1
            fixer += 1
    scores.sort(key=lambda x: x[2], reverse=False)
    count0 = 0
    for i in range(len(scores)):
        if count0 < k_0:
            scores[i][3] = 0
            count0 += 1
            fixer += 1
    
    m = grb.read(args.instance)
    m.Params.TimeLimit = 1000
    m.Params.Threads = 1
    m.Params.MIPFocus = 1
    m.Params.LogFile = 'gat'
    gurobi_log[m.Params.LogFile] = []
    
    instance_variabels = m.getVars()
    instance_variabels.sort(key=lambda v: v.VarName)
    variabels_map = {}
    for v in instance_variabels:  # get a dict (variable map), varname:var clasee
        variabels_map[v.VarName] = v
    alphas = []
    for i in range(len(scores)):
        tar_var = variabels_map[scores[i][1]]  # target variable <-- variable map
        x_star = scores[i][3]  # 1,0,-1, decide whether need to fix
        if x_star < 0:
            continue
        tmp_var = m.addVar(name=f'alp_{tar_var}', vtype=GRB.CONTINUOUS)
        alphas.append(tmp_var)
        m.addConstr(tmp_var >= tar_var - x_star, name=f'alpha_up_{i}')
        m.addConstr(tmp_var >= x_star - tar_var, name=f'alpha_dowm_{i}')
    all_tmp = 0
    for tmp in alphas:
        all_tmp += tmp
    m.addConstr(all_tmp <= delta, name="sum_alpha")
    
    m._starttime = time.monotonic()
    m.optimize(mycallback)
    
    with open('results/' + args.expname + '/gat/' + args.instance.split("/")[-1], "wb") as fp:
        pickle.dump(gurobi_log[m.Params.LogFile], fp)

    os.makedirs('results/' + args.expname + '/multitask/', exist_ok=True)
    
    saved_dict = torch.load(args.multitask_model, map_location=torch.device('cpu'))
    model = GATPolicy()
    model.load_state_dict(saved_dict)
    model.eval()
    
    #get bipartite graph as input
    A, v_map, v_nodes, c_nodes, b_vars=compute_mip_representation(args.instance)
    constraint_features = c_nodes.cpu()
    # constraint_features[np.isnan(constraint_features)] = 1 #remove nan value
    variable_features = v_nodes
    edge_indices = A._indices()
    edge_features = A._values().unsqueeze(1)
    edge_features=torch.ones(edge_features.shape)
    
    #prediction
    BD = model(
        constraint_features.to(DEVICE),
        edge_indices.to(DEVICE),
        edge_features.to(DEVICE),
        variable_features.to(DEVICE)
    ).sigmoid().cpu().squeeze()
    
    #align the variable name betweend the output and the solver
    all_varname=[]
    for name in v_map:
        all_varname.append(name)
    binary_name=[all_varname[i] for i in b_vars]
    scores=[]#get a list of (index, VariableName, Prob, -1, type)
    for i in range(len(v_map)):
        type="C"
        if all_varname[i] in binary_name:
            type='BINARY'
        scores.append([i, all_varname[i], BD[i].item(), -1, type])
    
    scores.sort(key=lambda x:x[2],reverse=True)
    
    scores=[x for x in scores if x[4]=='BINARY']#get binary
    
    fixer=0
    #fixing variable picked by confidence scores
    count1=0
    for i in range(len(scores)):
        if count1<k_1:
            scores[i][3] = 1
            count1+=1
            fixer += 1
    scores.sort(key=lambda x: x[2], reverse=False)
    count0 = 0
    for i in range(len(scores)):
        if count0 < k_0:
            scores[i][3] = 0
            count0 += 1
            fixer += 1
    
    m = grb.read(args.instance)
    m.Params.TimeLimit = 1000
    m.Params.Threads = 1
    m.Params.MIPFocus = 1
    m.Params.LogFile = 'multitask'
    gurobi_log[m.Params.LogFile] = []
    
    instance_variabels = m.getVars()
    instance_variabels.sort(key=lambda v: v.VarName)
    variabels_map = {}
    for v in instance_variabels:  # get a dict (variable map), varname:var clasee
        variabels_map[v.VarName] = v
    alphas = []
    for i in range(len(scores)):
        tar_var = variabels_map[scores[i][1]]  # target variable <-- variable map
        x_star = scores[i][3]  # 1,0,-1, decide whether need to fix
        if x_star < 0:
            continue
        tmp_var = m.addVar(name=f'alp_{tar_var}', vtype=GRB.CONTINUOUS)
        alphas.append(tmp_var)
        m.addConstr(tmp_var >= tar_var - x_star, name=f'alpha_up_{i}')
        m.addConstr(tmp_var >= x_star - tar_var, name=f'alpha_dowm_{i}')
    all_tmp = 0
    for tmp in alphas:
        all_tmp += tmp
    m.addConstr(all_tmp <= delta, name="sum_alpha")
    
    m._starttime = time.monotonic()
    m.optimize(mycallback)
    
    with open('results/' + args.expname + '/multitask/' + args.instance.split("/")[-1], "wb") as fp:
        pickle.dump(gurobi_log[m.Params.LogFile], fp)

    # os.makedirs('results/' + args.expname + '/multitask_v2/', exist_ok=True)
    
    # saved_dict = torch.load(args.multitask_model_v2, map_location=torch.device('cpu'))
    # model = GATPolicy()
    # model.load_state_dict(saved_dict)
    # model.eval()
    
    # #get bipartite graph as input
    # A, v_map, v_nodes, c_nodes, b_vars=compute_mip_representation(args.instance)
    # constraint_features = c_nodes.cpu()
    # # constraint_features[np.isnan(constraint_features)] = 1 #remove nan value
    # variable_features = v_nodes
    # edge_indices = A._indices()
    # edge_features = A._values().unsqueeze(1)
    # edge_features=torch.ones(edge_features.shape)
    
    # #prediction
    # BD = model(
    #     constraint_features.to(DEVICE),
    #     edge_indices.to(DEVICE),
    #     edge_features.to(DEVICE),
    #     variable_features.to(DEVICE)
    # ).sigmoid().cpu().squeeze()
    
    # #align the variable name betweend the output and the solver
    # all_varname=[]
    # for name in v_map:
    #     all_varname.append(name)
    # binary_name=[all_varname[i] for i in b_vars]
    # scores=[]#get a list of (index, VariableName, Prob, -1, type)
    # for i in range(len(v_map)):
    #     type="C"
    #     if all_varname[i] in binary_name:
    #         type='BINARY'
    #     scores.append([i, all_varname[i], BD[i].item(), -1, type])
    
    # scores.sort(key=lambda x:x[2],reverse=True)
    
    # scores=[x for x in scores if x[4]=='BINARY']#get binary
    
    # fixer=0
    # #fixing variable picked by confidence scores
    # count1=0
    # for i in range(len(scores)):
    #     if count1<k_1:
    #         scores[i][3] = 1
    #         count1+=1
    #         fixer += 1
    # scores.sort(key=lambda x: x[2], reverse=False)
    # count0 = 0
    # for i in range(len(scores)):
    #     if count0 < k_0:
    #         scores[i][3] = 0
    #         count0 += 1
    #         fixer += 1
    
    # m = grb.read(args.instance)
    # m.Params.TimeLimit = 1000
    # m.Params.Threads = 1
    # m.Params.MIPFocus = 1
    # m.Params.LogFile = 'multitask_v2'
    # gurobi_log[m.Params.LogFile] = []
    
    # instance_variabels = m.getVars()
    # instance_variabels.sort(key=lambda v: v.VarName)
    # variabels_map = {}
    # for v in instance_variabels:  # get a dict (variable map), varname:var clasee
    #     variabels_map[v.VarName] = v
    # alphas = []
    # for i in range(len(scores)):
    #     tar_var = variabels_map[scores[i][1]]  # target variable <-- variable map
    #     x_star = scores[i][3]  # 1,0,-1, decide whether need to fix
    #     if x_star < 0:
    #         continue
    #     tmp_var = m.addVar(name=f'alp_{tar_var}', vtype=GRB.CONTINUOUS)
    #     alphas.append(tmp_var)
    #     m.addConstr(tmp_var >= tar_var - x_star, name=f'alpha_up_{i}')
    #     m.addConstr(tmp_var >= x_star - tar_var, name=f'alpha_dowm_{i}')
    # all_tmp = 0
    # for tmp in alphas:
    #     all_tmp += tmp
    # m.addConstr(all_tmp <= delta, name="sum_alpha")
    
    # m._starttime = time.monotonic()
    # m.optimize(mycallback)
    
    # with open('results/' + args.expname + '/multitask_v2/' + args.instance.split("/")[-1], "wb") as fp:
    #     pickle.dump(gurobi_log[m.Params.LogFile], fp)