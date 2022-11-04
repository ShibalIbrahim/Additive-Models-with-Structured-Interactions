from __future__ import division
import sys
print(sys.version, sys.platform, sys.executable) #Displays what environment you are actually using.

from copy import deepcopy

import dill
import gc
from ipywidgets import *
import numpy as np
import pandas as pd
from scipy.special import comb

import pathlib
import argparse

sys.path.insert(0, os.path.abspath(str(pathlib.Path(__file__).absolute())).split('SparseAMsWithInteractions')[0])
from SparseAMsWithInteractions.src import data_utils
from SparseAMsWithInteractions.src import utils
from SparseAMsWithInteractions.src.AMsWithInteractionsStrongHierarchy import models

parser = argparse.ArgumentParser(description='Additive Models with Interactions under Strong Hierarchy on Census data.')

# Data Arguments
parser.add_argument('--load_directory', dest='load_directory',  type=str, default='/home/shibal/Census-Data')
parser.add_argument('--seed', dest='seed',  type=int, default=8)

# Logging Arguments
parser.add_argument('--version', dest='version',  type=int, default=1)

# Model Arguments
parser.add_argument('--Ki', dest='Ki',  type=int, default=10)
parser.add_argument('--Kij', dest='Kij',  type=int, default=5)
parser.add_argument('--relative_penalty', dest='r',  type=float, default=1.0)

# Algorithm Arguments
parser.add_argument('--max_interaction_support', dest='max_interaction_support',  type=int, default=400) # cuts off the L0 regularization path when the number of interactions reach 400.
parser.add_argument('--convergence_tolerance', dest='convergence_tolerance',  type=float, default=1e-4)
parser.add_argument('--grid_search', dest='grid_search',  type=str, default='full') # 'full', 'reduced'
# parser.add_argument('--load_model', dest='load_model', action='store_true')
# parser.add_argument('--no_load_model', dest='load_model', action='store_false') # only used when grid_search is 'reduced'
# parser.set_defaults(load_model=False)
parser.add_argument('--run_first_round', dest='run_first_round', action='store_true')
parser.add_argument('--no_run_first_round', dest='run_first_round', action='store_false')
parser.set_defaults(run_first_round=False)
parser.add_argument('--run_second_round', dest='run_second_round', action='store_true')
parser.add_argument('--no_run_second_round', dest='run_second_round', action='store_false')
parser.set_defaults(run_second_round=False)


# Tuning Arguments
parser.add_argument('--eval_criteria', dest='eval_criteria',  type=str, default='mse')

# Logging Arguments
parser.add_argument('--logging', dest='logging', action='store_true')
parser.add_argument('--no-logging', dest='logging', action='store_false')
parser.set_defaults(logging=True)

args = parser.parse_args()

# Import Processed Data

load_directory=args.load_directory
save_directory = os.path.join(os.path.abspath(str(pathlib.Path(__file__).absolute())).split('src')[0], "results") 

df_X, df_y, _ = data_utils.load_data(load_directory=load_directory,
                                  filename='pdb2019trv3_us.csv',
                                  remove_margin_of_error_variables=True)
seed = args.seed
np.random.seed(seed)
X, Y, Xval, Yval, Xtest, Ytest, _, y_scaler = data_utils.process_data(
    df_X,
    df_y,
    val_ratio=0.1, 
    test_ratio=0.1,
    seed=seed,
    standardize_response=False)

### Initialize parameters

convergence_tolerance = args.convergence_tolerance
column_names = df_X.columns
r = args.r
logging = args.logging
max_interaction_support=args.max_interaction_support
version = args.version

# for c in column_names:
#     print(c)

### How to run the model
path = os.path.join(save_directory, 'AMsWithInteractionsStrongHierarchy', 'v{}'.format(version), 'r{}'.format(r), 'seed{}'.format(seed))
os.makedirs(path, exist_ok=True)


with open(path+'/Parameters.txt', "w") as f:
    [f.write('{}: {}\n'.format(k,v)) for k,v in vars(args).items()]
    f.write('Train: {}, Validation: {}, Test: {}\n'.format(X.shape[0], Xval.shape[0], Xtest.shape[0])) 

eval_criteria = args.eval_criteria
p = X.shape[1]
N, _ = X.shape
Xmin = np.min(np.vstack([X, Xval, Xtest]), axis=0)
Xmax = np.max(np.vstack([X, Xval, Xtest]), axis=0)

###### Load fitted AMs with Interaction model and recover reduced supports for downstream Strong Hierarchy Model
load_path = os.path.join(save_directory, 'AMsWithInteractionsL0', 'v{}'.format(version), 'r{}'.format(r), 'seed{}'.format(seed), 'secondround')

with open(os.path.join(load_path, 'model_final.pkl'), 'rb') as input:
    ami = dill.load(input)
active_set = ami.active_set_union
interaction_terms = ami.interaction_terms_union[:10]
active_set = np.sort(np.union1d(active_set, np.unique(interaction_terms)))
# print("Number of main effects to consider:", len(active_set)) # we consider all main effects 
print("Number of interaction effects to consider:", len(interaction_terms))

###### Call and run AMs with Interaction with Strong Hierarchy Model    
# lams_sm_start = -3
# lams_sm_stop = -7
#lams_sm=np.logspace(start=lams_sm_start, stop=lams_sm_stop, num=20, base=10.0)
lams_sm = np.array([ami.lam_sm_opt])

lams_L0_start = 2
lams_L0_stop = -2
lams_L0 = np.logspace(start=lams_L0_start, stop=lams_L0_stop, num=1, base=10.0)
# lam_L0_index = 6 # optimal
# lams_L0 = [np.logspace(start=lams_L0_start, stop=lams_L0_stop, num=10, base=10.0)[lam_L0_index]]
amish = models.AMISH(lams_sm=lams_sm,
                     lams_L0=lams_L0,
                     convergence_tolerance=convergence_tolerance,
                     eval_criteria=eval_criteria,
                     path=path,
                     max_interaction_support=max_interaction_support,
                     terminate_val_L0path=False
                    )
amish.load_data(X, Y, y_scaler, column_names, Xmin, Xmax)
amish.get_main_and_interaction_set(interaction_terms=interaction_terms)
amish.generate_splines_and_quadratic_penalties(ami.Ki, ami.Kij)
amish.fitCV(Xval, Yval)
amish.evaluate_and_save(Xtest, Ytest)
amish.Btrain = None
amish.BtrainT_Btrain = None
amish.Btrain_interaction = None
amish.Btrain_interactionT_Btrain_interaction = None
with open(os.path.join(amish.path, 'model.pkl'), 'wb') as output:
    dill.dump(amish, output)