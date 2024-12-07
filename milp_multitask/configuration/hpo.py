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
)
from smac.utils.configspace import ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario

import pyscipopt
from pyscipopt import Model

def solve_milp(params=None, instance=""):
    model = Model()
    model.readProblem(instance)
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

    return sol, -primal, time
    
def _solve_(params, instance, seed):
    _, cost, _ = solve_milp(params, instance)
    return cost


def optimize(instance_file, runcount_limit=5, restore_incumbent=None, stats=None):
    # Configuration
    cs = ConfigurationSpace()
    params = [
        # Branching
        CategoricalHyperparameter(
            "branching/scorefunc", choices=["s", "p", "q"], default_value="p"
        ),
        UniformFloatHyperparameter("branching/scorefac", 0.0, 1.0, default_value=0.167),
        CategoricalHyperparameter(
            "branching/preferbinary", choices=[True, False], default_value=False
        ),
        UniformFloatHyperparameter("branching/clamp", 0.0, 0.5, default_value=0.2),
        UniformFloatHyperparameter("branching/midpull", 0.0, 1.0, default_value=0.75),
        UniformFloatHyperparameter("branching/midpullreldomtrig", 0.0, 1.0, default_value=0.5),
        CategoricalHyperparameter(
            "branching/lpgainnormalize", choices=["d", "l", "s"], default_value="s"
        ),
        # LP
        CategoricalHyperparameter(
            "lp/pricing", choices=["l", "a", "f", "p", "s", "q", "d"], default_value="l"
        ),
        UniformIntegerHyperparameter("lp/colagelimit", -1, 2147483647, default_value=10),
        UniformIntegerHyperparameter("lp/rowagelimit", -1, 2147483647, default_value=10),
        # Node Selection
        CategoricalHyperparameter(
            "nodeselection/childsel", choices=["d", "u", "p", "i", "l", "r", "h"], default_value="h"
        ),  # noqa
        # Separating
        UniformFloatHyperparameter("cutselection/hybrid/minortho", 0.0, 1.0, default_value=0.9),
        UniformFloatHyperparameter("cutselection/hybrid/minorthoroot", 0.0, 1.0, default_value=0.9),
        UniformIntegerHyperparameter("separating/maxcutsgenfactor", 0, 2147483647, default_value=100),
        UniformIntegerHyperparameter("separating/maxcutsrootgenfactor", 0, 2147483647, default_value=2000),
        UniformIntegerHyperparameter("separating/cutagelimit", -1, 2147483647, default_value=80),
        UniformIntegerHyperparameter("separating/poolfreq", -1, 65534, default_value=10),
    ]
    cs.add_hyperparameters(params)

    # Scenario object
    seed = np.random.randint(1000000, 9999999)
    scenario = Scenario(cs, deterministic=True, n_trials=runcount_limit, instances=[instance_file], seed=seed, output_directory='smac3_output/' + instance_file.split("/")[-1])

    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=_solve_,
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
    
    incumbent, configs = optimize(args.instance)
    os.makedirs("datasets/" + args.exp_name + "/", exist_ok=True)
    df = pd.DataFrame(configs)
    df.to_csv("datasets/" + args.exp_name + "/" + args.instance.split("/")[-1])