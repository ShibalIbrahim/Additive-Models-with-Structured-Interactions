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
import argparse
import pathlib
from sklearn.metrics import mean_squared_error
from IPython.display import display
from scipy import stats

sys.path.insert(0, os.path.abspath(str(pathlib.Path(__file__).absolute())).split('SparseAMsWithInteractions')[0])
from SparseAMsWithInteractions.src import data_utils
from SparseAMsWithInteractions.src import utils
from SparseAMsWithInteractions.src.AMsWithInteractionsL0 import models
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import KFold

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
             
parser = argparse.ArgumentParser(description='GAMI on synthetic data.')

# Data Arguments
parser.add_argument('--seed', dest='seed',  type=int, default=8)
parser.add_argument('--train_size', dest='train_size',  type=int, default=400)
parser.add_argument('--test_size', dest='test_size',  type=float, default=10000)
parser.add_argument('--dist', dest='dist',  type=str, default='normal')

# Logging Arguments
parser.add_argument('--version', dest='version',  type=int, default=1)
parser.add_argument('--r', dest='r',  type=float, default=2.0)


args = parser.parse_args()
                
      
np.random.seed(args.seed)

p = 10
Xtrain = np.random.random((args.train_size, p))
Xtest = np.random.random((args.test_size, p))

def g1(t):
    return t

def g2(t):
    return (2*t - 1)**2

def g3(t):
    return np.sin(2*np.pi*t)/(2-np.sin(2*np.pi*t))
                              
def g4(t):
    return 0.1*np.sin(2*np.pi*t)+0.2*np.cos(2*np.pi*t)+0.3*(np.sin(2*np.pi*t)**2)+0.4*(np.cos(2*np.pi*t)**3)+0.5*(np.sin(2*np.pi*t)**3)                          

def get_f(x):
    f = g1(x[:,0])+g2(x[:,1])+g3(x[:,2])+g4(x[:,3])+g1(x[:,2]*x[:,3])+g2(0.5*(x[:,0]+x[:,2]))+g3(x[:,0]*x[:,1])
    return f
ftrain = get_f(Xtrain)
ftest = get_f(Xtest)

if args.dist == 'normal':
    errortrain = np.random.normal(loc=0, scale=0.2546, size=ftrain.shape)
    errortest = np.random.normal(loc=0, scale=0.2546, size=ftest.shape)
elif args.dist == 'skewed':
    errortrain = stats.lognorm(s=0.2546, loc=-1.0).rvs(size=ftrain.shape)
    errortest = stats.lognorm(s=0.2546, loc=-1.0).rvs(size=ftest.shape)
elif args.dist == 'heteroskedastic':
    errortrain = np.random.normal(loc=0, scale=2*0.2546*g1(Xtrain[:,4]))
    errortest = np.random.normal(loc=0, scale=2*0.2546*g1(Xtest[:,4]))
else:
    raise ValueError(f"Error distribution {args.dist} is not supported")
    
ytrain = ftrain+errortrain
ytest = ftest+errortest
ytrain = ytrain.reshape(-1,1)
ytest = ytest.reshape(-1,1)   

def identity_func(x):
    return np.array(x)
y_preprocessor = FunctionTransformer(lambda x: np.array(x)) # acts as identity
y_scaler = y_preprocessor

save_directory = os.path.join("/pool001/shibal", "results-synthetic", args.dist, "N_train_{}".format(args.train_size), "seed{}".format(args.seed)) 

convergence_tolerance = 1e-4
column_names = np.arange(Xtrain.shape[1])
r = args.r
logging = True
max_interaction_support=400 # cuts off the L0 regularization path when the number of interactions reach 400.
version = args.version

num_of_folds = 5
kf = KFold(n_splits=num_of_folds, random_state=None)
kf.get_n_splits(Xtrain)

Ki = 10
Kij = 5
eval_criteria = 'mse'
_, p = Xtrain.shape
lams_sm_start = -2
lams_sm_stop = -6
lams_L0_start = -1
lams_L0_stop = -5

Xmin = np.min(np.vstack([Xtrain, Xtest]), axis=0)
Xmax = np.max(np.vstack([Xtrain, Xtest]), axis=0)
lams_sm=np.logspace(start=lams_sm_start, stop=lams_sm_stop, num=20, base=10.0)
lams_L0=np.logspace(start=lams_L0_start, stop=lams_L0_stop, num=50, base=10.0)


path = os.path.join(
    save_directory,
    'AMsWithInteractionsL0',
    'v{}'.format(version),
    'r{}'.format(r),
)
os.makedirs(path, exist_ok=True)

for fold, (train_index, val_index) in enumerate(kf.split(Xtrain)):
    print("===================FOLD: {} ================".format(fold))
#     print("TRAIN:", train_index, "VAL:", val_index)
    X_train, X_val = Xtrain[train_index], Xtrain[val_index]
    y_train, y_val = ytrain[train_index], ytrain[val_index]
    
    path_fold = os.path.join(path,'fold{}'.format(fold))
    os.makedirs(path_fold, exist_ok=True)
    
    with open(path_fold+'/Parameters.txt', "w") as f:
        f.write('Logging: {}, tol: {:.6f}, Train: {}, Validation: {}, Test: {}\n'.format(
            logging, convergence_tolerance, X_train.shape[0], X_val.shape[0], Xtest.shape[0]))

        
    am = models.AMI(lams_sm=lams_sm,
                    lams_L0=lams_L0,
                    alpha=r,
                    convergence_tolerance=convergence_tolerance,
                    eval_criteria=eval_criteria,
                    path=path_fold,
                    max_interaction_support=max_interaction_support,
                    terminate_val_L0path=False
                    )
    am.load_data(X_train, y_train, y_scaler, column_names, Xmin, Xmax)
    am.generate_interaction_terms(generate_all_pairs=True)
    print("Number of interaction effects to consider:", len(am.interaction_terms))
    
    am.generate_splines_and_quadratic_penalties(Ki=Ki, Kij=Kij)
    am.fitCV(X_val, y_val)
    
    interaction_terms = np.array([am.interaction_terms[k] for k in am.active_interaction_set_opt])
    
    if fold==0:
        print("=========interaction_terms:", interaction_terms)
        interaction_terms_union = deepcopy(interaction_terms)
        print("=========interaction_terms_union:", interaction_terms_union)
    else:
        print("=========interaction_terms:", interaction_terms)
        if len(interaction_terms)>0 and len(interaction_terms_union)>0:
            interaction_terms_union = np.concatenate([interaction_terms_union, interaction_terms], axis=0)
        elif len(interaction_terms)==0:
            pass
        elif len(interaction_terms_union)==0:
            interaction_terms_union = deepcopy(interaction_terms)
        print("=========interaction_terms_union:", interaction_terms_union)

    am.evaluate_and_save(Xtest, ytest)
    am.Btrain = None
    am.BtrainT_Btrain = None
    am.Btrain_interaction = None
    am.Btrain_interactionT_Btrain_interaction = None
    with open(os.path.join(path_fold, 'model.pkl'), 'wb') as output:
        dill.dump(am, output)
        
    with open(path_fold+'/Results.txt', "a") as f:
        f.write('Main-effects: {}\n'.format(am.active_set_opt))
        f.write('Interaction-effects: {}\n'.format([am.interaction_terms[k] for k in am.active_interaction_set_opt]))



###### Read csv files per fold to find optimal hyperparameters        
for fold in range(num_of_folds):
    print(fold)
    df_temp = pd.read_csv(os.path.join(os.path.join(path,'fold{}'.format(fold)), 'Training.csv')).set_index(['Smoothness', '        L0      '])[['    val     ']]
    df_temp.columns = ['val-{}'.format(fold)]
    if fold==0:
        df = df_temp.copy()
    else:
        df = df.join(df_temp, how='outer')    
    display(df)
dfr = df.reset_index()
dfr = dfr.sort_values(by=['Smoothness'], ascending=False).set_index(['Smoothness', '        L0      '])
dfr = dfr.mean(axis=1)
dfr = dfr[dfr==dfr.min()].reset_index()        
display(dfr)
L0_opt = dfr['        L0      '].values[0]
Smoothness_opt = dfr['Smoothness'].values[0]   


###### Refit for optimal smoothness on train + val
with open(path+'/Parameters.txt', "w") as f:
    f.write('Logging: {}, tol: {:.6f}, Train+Validation: {}, Test: {}\n'.format(
        logging, convergence_tolerance, Xtrain.shape[0], Xtest.shape[0]))

lams_sm_opt=np.array([Smoothness_opt])
lams_L0_opt=np.array([L0_opt]) 
# lams_L0_opt=lams_L0[lams_L0>=L0_opt]

am = models.AMI(lams_sm=lams_sm_opt,
                lams_L0=lams_L0_opt,
                alpha=r,
                convergence_tolerance=convergence_tolerance,
                eval_criteria=eval_criteria,
                path=path,
                max_interaction_support=max_interaction_support,
                terminate_val_L0path=False
                )
am.load_data(Xtrain, ytrain, y_scaler, column_names, Xmin, Xmax)

if len(interaction_terms_union)>0:
    interaction_terms_union = np.unique(interaction_terms_union, axis=0)
    am.interaction_terms = interaction_terms_union
    am.generate_all_pairs = False
    am.I = (int)(comb(am.p, 2, exact=False))
    am.Imax = len(interaction_terms_union)
else:
    am.generate_interaction_terms(generate_all_pairs=True)

print("Number of interaction effects to consider:", len(am.interaction_terms))

am.generate_splines_and_quadratic_penalties(Ki=Ki, Kij=Kij)
am.fitCV(Xval=Xtest, Yval=ytest)

am.evaluate_and_save(Xtest, ytest)
am.Btrain = None
am.BtrainT_Btrain = None
am.Btrain_interaction = None
am.Btrain_interactionT_Btrain_interaction = None
with open(os.path.join(path, 'model.pkl'), 'wb') as output:
    dill.dump(am, output)


ftest_predict = am.predict(Xtest)
true_error = mean_squared_error(ftest, ftest_predict)

main_effects = np.array(am.active_set_opt)
interaction_effects = np.array([am.interaction_terms[k] for k in am.active_interaction_set_opt])

# Compute FPR and FNR for main effects
main_support_true = np.array([1,1,1,1,0,0,0,0,0,0])
main_support_recovered = np.zeros_like(main_support_true)
main_support_recovered[main_effects] = 1
tpr_main = recall_score(main_support_true, main_support_recovered)   # it is better to name it y_test 
# to calculate, tnr we need to set the positive label to the other class
# I assume your negative class consists of 0, if it is -1, change 0 below to that value
tnr_main = recall_score(main_support_true, main_support_recovered, pos_label=0) 
fpr_main = 1 - tnr_main
fnr_main = 1 - tpr_main   
f1_main = f1_score(main_support_true, main_support_recovered)

# Compute FPR and FNR for interaction effects
interaction_terms = []
for m in range(0, p):
    for n in range(0, p):
        if m!=n and m<n:
            interaction_terms.append((m, n))
interaction_terms = np.array(interaction_terms)
interaction_support_recovered = np.zeros((len(interaction_terms)))
if len(interaction_effects)>0:
    for term in interaction_effects:
        interaction_support_recovered[(term.reshape(1,-1)==interaction_terms).all(axis=1)] = 1

interaction_support_true = np.zeros((len(interaction_terms)))
for term in np.array([[0,1],[0,2],[2,3]]):
    interaction_support_true[(term.reshape(1,-1)==interaction_terms).all(axis=1)] = 1

from sklearn.metrics import recall_score
tpr_interaction = recall_score(interaction_support_true, interaction_support_recovered)   # it is better to name it y_test 
# to calculate, tnr we need to set the positive label to the other class
# I assume your negative class consists of 0, if it is -1, change 0 below to that value
tnr_interaction = recall_score(interaction_support_true, interaction_support_recovered, pos_label=0) 
fpr_interaction = 1 - tnr_interaction
fnr_interaction = 1 - tpr_interaction    
f1_interaction = f1_score(interaction_support_true, interaction_support_recovered)


with open(path+'/Results.txt', "a") as f:
    f.write('\n True Test MSE: {}\n'.format(true_error))
    f.write('FPR (main): {}\n'.format(fpr_main))
    f.write('FNR (main): {}\n'.format(fnr_main))
    f.write('F1 (main): {}\n'.format(f1_main))
    f.write('FPR (interactions): {}\n'.format(fpr_interaction))
    f.write('FNR (interactions): {}\n'.format(fnr_interaction))
    f.write('F1 (interactions): {}\n'.format(f1_interaction))
    f.write('Main-effects: {}\n'.format(main_effects))
    f.write('Interaction-effects: {}\n'.format(interaction_effects))

with open(path+'/support_set.npy', 'wb') as f:
    np.save(f, main_effects)
    np.save(f, interaction_effects)