import os

import torch
import torch_geometric
import gzip
import pickle
import random
import numpy as np
import time
import pyscipopt as scp
import gurobipy as gp
from IPython import embed
import torch.nn.init as init

import datetime as dt
from pathlib import Path
import logging

from tqdm import tqdm
import pandas as pd
import sklearn
import sklearn.metrics
import numpy as np
import gurobipy as grb
import pickle
import torch
import torch_geometric as tg
from ast import literal_eval
from copy import deepcopy
import re

from sklearn.metrics.pairwise import cosine_similarity

variable_feature_names = [
    "type_CONTINUOUS",
    "type_BINARY",
    "type_INTEGER",
    "coef",
    "has_lb",
    "has_ub",
    "sol_is_at_lb",
    "sol_is_at_ub",
    "sol_frac",
    "basis_status_BASIC",
    "basis_status_NONBASIC_LOWER",
    "basis_status_NONBASIC_UPPER",
    "basis_status_SUPERBASIC",
    "reduced_cost",
    "sol_val"
    ]

constraint_feature_names = [
    "obj_cos_sim",
    "bias",
    "is_tight",
    "dualsol_val",
]

num_var_features = len(variable_feature_names)
num_con_features = len(constraint_feature_names)
env = grb.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()

def normalize_features(data):
    # data.x_vars = # normalize this
    # data.x_cons
    var_scaler = sklearn.preprocessing.StandardScaler()
    con_scaler = sklearn.preprocessing.StandardScaler()
    data.x_vars = torch.tensor(var_scaler.fit_transform(data.x_vars.numpy()))
    data.x_cons = torch.tensor(con_scaler.fit_transform(data.x_cons.numpy()))
    return data

def compute_fractionality(mip_file, seed, presolve=False):
    """Given a mip instance and a presolve flag,
       return the indices to integer vars in the (presolved) model,
       fractionality of the integer variables,
       (presolved) model."""

    model = grb.read(mip_file)

    if presolve:
        model = model.presolve()

    model.setParam("Seed", seed)

    relaxation = model.relax()
    relaxation.optimize()

    fractions = []
    discrete_vars = [i for i, x in enumerate(model.getVars())
                     if x.vType !=grb.GRB.CONTINUOUS]
    all_relax_vars = relaxation.getVars()

    assert(len(model.getVars()) == len(all_relax_vars))

    for i in discrete_vars:
        relax_var = all_relax_vars[i]
        fractions.append(abs(round(relax_var.x) - relax_var.x))

    return discrete_vars, np.array(fractions), model

def compute_mip_representation(mip_file):
    """given a mip instance, compute features and return pytorch_geometric data instance describing variable constraint graph

    Args:
        mip_file (str): mip instance file location, readable by gurobi
    """
    model = gp.read(str(mip_file))
    A = model.getA()
    objective_coefficients = np.array([x.Obj for x in model.getVars()])
    relaxation = model.relax()
    relaxation.optimize()

    discrete_var_mask = torch.zeros(len(model.getVars()), dtype=torch.bool)

    # compute variable features
    # collect into list of variable features i.e. num_vars x num_var_features sized matrix X_v
    variable_features = []
    v_map = {}
    b_vars = []
    for var_ind, (decision_var, relax_var) in enumerate(zip(model.getVars(), relaxation.getVars())):
        feature_vector = [
            decision_var.VType == gp.GRB.CONTINUOUS,
            decision_var.VType == gp.GRB.BINARY,
            decision_var.VType == gp.GRB.INTEGER,
            decision_var.Obj,
            decision_var.LB > -gp.GRB.INFINITY,
            decision_var.UB <= gp.GRB.INFINITY,
            relax_var.x == relax_var.LB,
            relax_var.x == relax_var.UB,
            abs(round(relax_var.x) - relax_var.x),
            relax_var.VBasis == gp.GRB.BASIC,
            relax_var.VBasis == gp.GRB.NONBASIC_LOWER,
            relax_var.VBasis == gp.GRB.NONBASIC_UPPER,
            relax_var.VBasis == gp.GRB.SUPERBASIC,
            relax_var.RC,
            relax_var.x
        ]
        v_map[decision_var.varname] = var_ind
        if decision_var.VType == gp.GRB.BINARY:
            b_vars.append(var_ind)
        discrete_var_mask[var_ind] = decision_var.VType != gp.GRB.CONTINUOUS
        variable_features.append(feature_vector)

    # compute constraint features
    # collect into list of constraint features i.e. num_cons x num_con_features sized matrix X_c
    cosine_sims = cosine_similarity(A, objective_coefficients.reshape(1, -1))
    constraint_features = []
    for con_ind, (con, relax_con) in enumerate(zip(model.getConstrs(), relaxation.getConstrs())):
        feature_vector = [
            cosine_sims[con_ind, 0],
            con.RHS,
            relax_con.Slack == 0,
            relax_con.Pi
        ]
        constraint_features.append(feature_vector)
    # compute edge features
    # collect list of edge features i.e. num_edges x num_edge_features
    edge_features = []
    edge_indices = np.array(A.nonzero())
    # length num edges long vector of features, just contains the nonzero coeffs for now
    edge_features = np.array(A[edge_indices[0], edge_indices[1]].T).reshape(-1)

    A = torch.sparse_coo_tensor(edge_indices, edge_features, (len(model.getConstrs()), len(model.getVars())))

    variable_features = torch.as_tensor(variable_features, dtype=torch.float32)
    constraint_features = torch.as_tensor(constraint_features, dtype=torch.float32)
    b_vars = torch.as_tensor(b_vars, dtype=torch.int32)

    return A, v_map, variable_features, constraint_features, b_vars

class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
            self,
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,

    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features



    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

def get_perturbed_solution(instance_name, sol, varNames, varname_map, b_vars, m_tmp, init_success_count = 0):
    #m = scp.Model()
    #m.readProblem(instance_name)
    
    #mvars = m.getVars()
    #mvars.sort(key=lambda v: v.name)
    success_count = init_success_count 
    example_limit = 10
    perturbation_rate = 0.1#0.05
    negative_samples = []
    #m_tmp = scp.Model()
    #m_tmp.hideOutput(True)
    #m_tmp.readProblem(instance_name)
    vars = m_tmp.getVars()
    var_map = {}
    vars.sort(key=lambda v: v.name)
    for _i, varName in enumerate(varNames):  # get a dict (variable map), varname:var clasee
        var_map[varName] = _i

    while perturbation_rate <= 1 and success_count <= example_limit:
        for i in range(example_limit):
            selected = set(random.sample(list(range(b_vars.shape[0])), int(b_vars.shape[0] * perturbation_rate)))
            
            if "load_balancing" in instance_name:
                pass
            else:
                m_tmp = scp.Model()
                m_tmp.hideOutput(True)
                m_tmp.readProblem(instance_name)
                
                vars = m_tmp.getVars()
                var_map = {}
                vars.sort(key=lambda v: v.name)
                for _i, varName in enumerate(varNames):  # get a dict (variable map), varname:var clasee
                    var_map[varName] = _i

            sample = [0] * m_tmp.getNVars()
            for j in range(b_vars.shape[0]):
                
                v = vars[b_vars[j]]
                sol_index = var_map[v.name]
                if j in selected:
                    sample[sol_index] = 1 - sol[sol_index]
                else:
                    #m_tmp.fixVar(var_map[varNames[b_vars[j]]], sol[b_vars[j]])
                    sample[sol_index] = sol[sol_index]
                #embed()
                #v = var_map[varNames[b_vars[j]]]
                #v = vars[b_vars[j]]

                if "load_balancing" in instance_name: # just do random pertubation to save time, it is infeasible with very high probability
                    continue

                
                fixed_value = sample[sol_index]
                assert fixed_value in [0,1], f"variable {v.name} with type {v.vtype()} fixed to value {fixed_value}"
                m_tmp.chgVarLb(v, fixed_value)
                m_tmp.chgVarLbGlobal(v, fixed_value)
                m_tmp.chgVarUb(v, fixed_value)
                m_tmp.chgVarUbGlobal(v, fixed_value)
                if v.vtype() != "BINARY":
                    embed()
                #embed();assert False

                assert v.vtype() == "BINARY", f"fixing a non-binary variable {v.name} with type {v.vtype()}"

            #m_tmp.setParam("limits/solutions",100)
            if "load_balancing" in instance_name:
                solution_count = 0
            else:   
                m_tmp.setParam("limits/time", 60)
                #m_tmp.setParam("constraints/countsols/sollimit",1)
                m_tmp.setParam("limits/solutions",1)
                m_tmp.optimize()
                solution_count = m_tmp.getNSols()
            #embed();exit()
            if "298" in instance_name and "MVC" in instance_name:
                embed()
            if solution_count == 0: # infeasible solution:
                success_count += 1
                negative_samples.append(sample)
                if success_count >= example_limit: 
                    break
                print(f"Succeed with pertubation rate {perturbation_rate} {i}; Success count: {success_count}")
            else:
                sol_tmp = m_tmp.getBestSol()
                obj_tmp = m_tmp.getSolObjVal(sol_tmp)
                print(f"failed with pertubation rate {perturbation_rate} {i} obj_value = {obj_tmp}")
                embed();exit()
                pass
        perturbation_rate += 0.05
    return negative_samples
        

def generate_neg_samples(instance_name, sols, varNames, varname_map, b_vars, args=None, filename=None):
    
    neg_examples = []

    solData = None
    if not (filename is None):
        #if not (input_args is None) and input_args.GNN == "GAT":
        #    BGFilepath += "GAT"
        BGFilepath, solFilePath = filename
        with open(solFilePath, "rb") as f:
            solData = pickle.load(f)

    example_limit = 10
    for i in range(sols.shape[0]):
        neg_ex = []
        if not (solData is None):
            if "item" in solFilePath:
                neg_ex_priority_list = ["neg_examples_iLB_9", "neg_examples_iLB_0", "neg_examples_iLB_1"]#, "neg_examples_perturb_0","neg_examples_perturb_1"]
                #print("ilb9")
            else:
                neg_ex_priority_list = ["neg_examples_iLB_0", "neg_examples_iLB_1"]#, "neg_examples_perturb_0","neg_examples_perturb_1"]
            if args.negex == "perturb":
                neg_ex_priority_list = ["neg_examples_perturb_0","neg_examples_perturb_1"]
            #print(neg_ex_priority_list)
            for entry in neg_ex_priority_list:
                if not (entry in solData): continue
                if "perturb" in entry and args.negex != "perturb":
                    print("required perturb", -len(neg_ex) + example_limit)
                entry_neg_ex = solData[entry][i][0].tolist()
                if not (entry_neg_ex is None) and len(entry_neg_ex) > 0:
                    neg_ex+= entry_neg_ex[:example_limit - len(neg_ex)]
                if len(neg_ex) >= example_limit:
                    break
        #embed()
        if len(neg_ex) < example_limit:
            print("don't have sufficient neg examples, collecting for", instance_name)
            m_tmp = scp.Model()
            m_tmp.hideOutput(True)
            m_tmp.readProblem(instance_name)
            neg_ex += get_perturbed_solution(instance_name, sols[i], varNames, varname_map, b_vars, m_tmp, init_success_count=len(neg_ex))
        #embed();exit()
        neg_examples += neg_ex
    return neg_examples


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files, input_args):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.negative_examples_dict = {}
        self.input_args = input_args
        # Preload the data into memory during initialization
        self.data = [self._load_graph(index) for index in range(len(self.sample_files))]
        

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        Return the graph data that has been preloaded into memory.
        """
        return self.data[index]

    def _load_graph(self, index):
        """
        Loads the graph data from disk and prepares it.
        """
        inspath, resPath = self.sample_files[index]

        A, v_map, v_nodes, c_nodes, b_vars=compute_mip_representation(inspath)

        constraint_features = c_nodes
        edge_indices = A._indices()

        variable_features = v_nodes
        edge_features =A._values().unsqueeze(1)
        edge_features=torch.ones(edge_features.shape)

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
        )

        with open(resPath, "rb") as f:
            solData = pickle.load(f)

        varNames = solData['var_names']
        sols = solData['sols']
        objs = solData['objs']
        pd_gap = solData['pd_gap']
        instance_name = inspath
        sols=np.round(sols,0)
        
        negative_sample = sols[max(sols.shape[0]-350,50):]
        negative_sample_objective = objs[max(objs.shape[0]-350,50):]
        sols = sols[:50]
        objs = objs[:50]
        
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.solutions = torch.FloatTensor(sols).reshape(-1)

        graph.objVals = torch.FloatTensor(objs)
        graph.nsols = sols.shape[0]
        graph.ntvars = variable_features.shape[0]
        graph.varNames = varNames
        varname_dict={}
        varname_map=[]
        i=0
        for iter in varNames:
            varname_dict[iter]=i
            i+=1
        for iter in v_map:
            varname_map.append(varname_dict[iter])

        varname_map=torch.tensor(varname_map)

        graph.varInds = [[varname_map],[b_vars]]

        graph.pd_gap = pd_gap
        graph.instance_name = instance_name

        if not (instance_name in self.negative_examples_dict):
            self.negative_examples_dict[instance_name] = np.array(generate_neg_samples(instance_name, sols, varNames, varname_map, b_vars, args=self.input_args, filename=self.sample_files[index]))
            print("Finish processing data for", instance_name)

        negative_samples = self.negative_examples_dict[instance_name]
    
        graph.nnegsamples = negative_samples.shape[0]
        graph.negsamples = torch.FloatTensor(negative_samples)

        return graph