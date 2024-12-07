import os
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
    model = grb.read(str(mip_file), env=env)
    A = model.getA()
    objective_coefficients = np.array([x.Obj for x in model.getVars()])
    relaxation = model.relax()
    relaxation.optimize()

    discrete_var_mask = torch.zeros(len(model.getVars()), dtype=torch.bool)

    # compute variable features
    # collect into list of variable features i.e. num_vars x num_var_features sized matrix X_v
    variable_features = []
    for var_ind, (decision_var, relax_var) in enumerate(zip(model.getVars(), relaxation.getVars())):
        feature_vector = [
            decision_var.VType == grb.GRB.CONTINUOUS,
            decision_var.VType == grb.GRB.BINARY,
            decision_var.VType == grb.GRB.INTEGER,
            decision_var.Obj,
            decision_var.LB > -grb.GRB.INFINITY,
            decision_var.UB <= grb.GRB.INFINITY,
            relax_var.x == relax_var.LB,
            relax_var.x == relax_var.UB,
            abs(round(relax_var.x) - relax_var.x),
            relax_var.VBasis == grb.GRB.BASIC,
            relax_var.VBasis == grb.GRB.NONBASIC_LOWER,
            relax_var.VBasis == grb.GRB.NONBASIC_UPPER,
            relax_var.VBasis == grb.GRB.SUPERBASIC,
            relax_var.RC,
            relax_var.x
        ]
        discrete_var_mask[var_ind] = decision_var.VType != grb.GRB.CONTINUOUS
        variable_features.append(feature_vector)

    # compute constraint features
    # collect into list of constraint features i.e. num_cons x num_con_features sized matrix X_c
    cosine_sims =sklearn.metrics.pairwise.cosine_similarity(A, objective_coefficients.reshape(1, -1))
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
    edge_features = A[edge_indices[0], edge_indices[1]].T

    # get mip features in graph
    data = BipartiteNodeData(
        x_vars = torch.tensor(np.array(variable_features), dtype=torch.float),
        x_var_names = variable_feature_names,
        x_cons = torch.tensor(np.array(constraint_features), dtype=torch.float),
        x_con_names = constraint_feature_names,
        edge_index_cons_to_vars = torch.tensor(edge_indices, dtype=torch.long),
        edge_index_var_to_cons = torch.tensor(edge_indices[::-1].copy(), dtype=torch.long),
        edge_attr = torch.tensor(edge_features, dtype=torch.float),
        # edge_index = torch.tensor(edge_indices, dtype=torch.long),
        # edge_attr = torch.tensor(edge_features),
        discrete_var_mask = discrete_var_mask
    )

    return data

class BipartiteNodeData(tg.data.Data):
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_cons_to_vars':
            return torch.tensor([[self.x_cons.size(0)], [self.x_vars.size(0)]])
        elif key == "edge_index_var_to_cons":
            return torch.tensor([[self.x_vars.size(0)], [self.x_cons.size(0)]])
        else:
            return super(BipartiteNodeData, self).__inc__(key, value)
        
class GraphDataset(tg.data.InMemoryDataset):
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.negative_examples_dict = {}
        # Preload the data into memory during initialization
        self.data = [self._load_graph(index) for index in range(len(self.sample_files))]

    def len(self):
        return len(self.data)

    def get(self, index):
        """
        Return the graph data that has been preloaded into memory.
        """
        return self.data[index]

    def _load_graph(self, index):
        """
        Loads the graph data from disk and prepares it.
        """
        insPath, resPath = self.sample_files[index]
        print(f"Loading graph from: {insPath}")

        graph = compute_mip_representation(insPath)

        df = pd.read_csv(resPath, index_col=0)

        graph.pos_sample = []
        graph.neg_sample = []

        for i in range(5):
            one_hot_backdoor = np.zeros(graph.x_vars.shape[0])
            for var_ind in literal_eval(df.iloc[i]["backdoor_list"]):
                one_hot_backdoor[int(var_ind)] = 1
            graph.pos_sample.append(one_hot_backdoor)
            one_hot_backdoor = np.zeros(graph.x_vars.shape[0])
            for var_ind in literal_eval(df.iloc[-i-1]["backdoor_list"]):
                one_hot_backdoor[int(var_ind)] = 1
            graph.neg_sample.append(one_hot_backdoor)

        return graph

class GraphDataset_filter(tg.data.InMemoryDataset):
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.negative_examples_dict = {}
        # Preload the data into memory during initialization
        self.data = [self._load_graph(index) for index in range(len(self.sample_files))]

    def len(self):
        return len(self.data)

    def get(self, index):
        """
        Return the graph data that has been preloaded into memory.
        """
        return self.data[index]

    def _load_graph(self, index):
        """
        Loads the graph data from disk and prepares it.
        """
        insPath, resPath = self.sample_files[index]
        print(f"Loading graph from: {insPath}")

        graph = compute_mip_representation(insPath)

        df = pd.read_csv(resPath, index_col=0)

        graph.pos_sample = []
        graph.neg_sample = []

        for i in range(5):
            one_hot_backdoor = np.zeros(graph.x_vars.shape[0])
            for var_ind in literal_eval(df.iloc[i]["backdoor_list"]):
                one_hot_backdoor[int(var_ind)] = 1
            if df.iloc[i]["run_time"] <= 1:
                graph.pos_sample.append(one_hot_backdoor)
            one_hot_backdoor = np.zeros(graph.x_vars.shape[0])
            for var_ind in literal_eval(df.iloc[-i-1]["backdoor_list"]):
                one_hot_backdoor[int(var_ind)] = 1
            if df.iloc[-i-1]["run_time"] >= 1:
                graph.neg_sample.append(one_hot_backdoor)

        return graph