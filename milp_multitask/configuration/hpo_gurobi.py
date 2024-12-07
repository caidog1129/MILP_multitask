import os
nthreads = 1
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)
import argparse

import numpy as np
import pandas as pd
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    OrdinalHyperparameter,
)
from smac.utils.configspace import ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario

import gurobipy as grb
from gurobipy import GRB

env = grb.Env(empty=True)
env.setParam("OutputFlag",0)
env.setParam("Threads",1)
env.start()

def solve_milp(params=None, instance=""):
    model = grb.read(instance, env=env)
    model.setParam("LogToConsole", 0)
    model.setParam("TimeLimit", 15 * 60)

    if params:
        params = {k: params[k] for k in params}
        print(params)
        for param, value in params.items():
            model.setParam(param, value)

    model.optimize()

    primal = model.ObjVal if model.SolCount > 0 else float("inf")
    dual = model.ObjBound
    time = model.Runtime

    return None, -primal, time

def _solve_(params, instance, seed):
    _, cost, _ = solve_milp(params, instance)
    return cost

def _solve2_(params, instance, seed):
    _, cost, _ = solve_milp(params, instance)
    return -cost

def optimize(instance_file, exp_name, runcount_limit=50, restore_incumbent=None, stats=None):
    # Configuration
    cs = ConfigurationSpace()
    params = [
        # Add Gurobi-specific parameters
        CategoricalHyperparameter("BranchDir", choices=[-1, 0, 1], default_value=0),
        UniformIntegerHyperparameter("DegenMoves", -1, 2000000000, default_value=-1),
        CategoricalHyperparameter("Disconnected", choices=[-1, 0, 1, 2], default_value=-1),
        UniformFloatHyperparameter("Heuristics", 0.0, 1.0, default_value=0.05),
        CategoricalHyperparameter("NodeMethod", choices=[-1, 0, 1, 2], default_value=-1),
        CategoricalHyperparameter("Method", choices=[-1, 0, 1, 2, 3, 4, 5], default_value=-1),
        UniformIntegerHyperparameter("PartitionPlace", 0, 31, default_value=15),
        CategoricalHyperparameter("Symmetry", choices=[-1, 0, 1, 2], default_value=-1),
        CategoricalHyperparameter("VarBranch", choices=[-1, 0, 1, 2, 3], default_value=-1),
        CategoricalHyperparameter("Cuts", choices=[-1, 0, 1, 2, 3], default_value=-1),
        CategoricalHyperparameter("CliqueCuts", choices=[-1, 0, 1, 2], default_value=-1),
        CategoricalHyperparameter("CoverCuts", choices=[-1, 0, 1, 2], default_value=-1),
        CategoricalHyperparameter("FlowCoverCuts", choices=[-1, 0, 1, 2], default_value=-1),
        CategoricalHyperparameter("FlowPathCuts", choices=[-1, 0, 1, 2], default_value=-1),
        CategoricalHyperparameter("GUBCoverCuts", choices=[-1, 0, 1, 2], default_value=-1),
        CategoricalHyperparameter("ImpliedCuts", choices=[-1, 0, 1, 2], default_value=-1),
        CategoricalHyperparameter("MIPSepCuts", choices=[-1, 0, 1, 2], default_value=-1),
        CategoricalHyperparameter("MIRCuts", choices=[-1, 0, 1, 2], default_value=-1),
        CategoricalHyperparameter("MIPFocus", choices=[0, 1, 2, 3], default_value=0),
        CategoricalHyperparameter("StrongCGCuts", choices=[-1, 0, 1, 2], default_value=-1),
        CategoricalHyperparameter("ModKCuts", choices=[-1, 0, 1, 2], default_value=-1),
        CategoricalHyperparameter("NetworkCuts", choices=[-1, 0, 1, 2], default_value=-1),
        CategoricalHyperparameter("ProjImpliedCuts", choices=[-1, 0, 1, 2], default_value=-1),
        CategoricalHyperparameter("SubMIPCuts", choices=[-1, 0, 1, 2], default_value=-1),
        CategoricalHyperparameter("ZeroHalfCuts", choices=[-1, 0, 1, 2], default_value=-1),
        CategoricalHyperparameter("InfProofCuts", choices=[-1, 0, 1, 2], default_value=-1),
        UniformIntegerHyperparameter("CutPasses", -1, 2000000000, default_value=-1),
        UniformIntegerHyperparameter("GomoryPasses", -1, 2000000000, default_value=-1),
        OrdinalHyperparameter("ImproveStartGap", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], default_value=0),
        UniformIntegerHyperparameter("RINS", -1, 2000000000, default_value=-1),
        UniformIntegerHyperparameter("SubMIPNodes", -1, 2000000000, default_value=500),
        CategoricalHyperparameter("Presolve", choices=[-1, 0, 1, 2], default_value=-1),
        UniformIntegerHyperparameter("PrePasses", -1, 2000000000, default_value=-1),
        CategoricalHyperparameter("Aggregate", choices=[0, 1], default_value=1),
        UniformIntegerHyperparameter("AggFill", -1, 2000000000, default_value=-1),
        CategoricalHyperparameter("PreSparsify", choices=[-1, 0, 1], default_value=-1),
    ]
    cs.add_hyperparameters(params)

    # Scenario object
    seed = np.random.randint(1000000, 9999999)
    scenario = Scenario(cs, deterministic=True, n_trials=runcount_limit, instances=[instance_file], seed=seed, output_directory='smac3_output/' + instance_file.split("/")[-1])

    
    if "CA" in exp_name or "INDSET" in exp_name or "IS" in exp_name or "MIS" in exp_name:
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=_solve_,
        )
    else:
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=_solve2_,
        )

    incumbent = smac.optimize()

    configs = []
    for key, value in smac.runhistory._data.items():  # noqa
        tmp = []
        tmp.append(value.cost)
        tmp.append(smac.runhistory.ids_config[key.config_id]._values)
        configs.append(tmp)
        
    return incumbent, configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance")
    parser.add_argument("--exp_name")
    args = parser.parse_args()
    
    incumbent, configs = optimize(args.instance, args.exp_name)
    os.makedirs("datasets/" + args.exp_name + "/", exist_ok=True)
    df = pd.DataFrame(configs)
    df.to_csv("datasets/" + args.exp_name + "/" + args.instance.split("/")[-1])
