import numpy as np
from copy import deepcopy
from tqdm import tnrange, tqdm_notebook
from IPython.display import display
from IPython.display import Math
from ipywidgets import *
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.special import comb
import pandas as pd

def CrossValidation(L0path_HS = None,
                    MIP_HS = None,
                    beta = None,
                    zeta = None,
                    delta = None,
                    alpha = None,
                    B = None,
                    B_interaction = None,
                    interaction_terms = None,
                    column_names = None,
                    lams_1 = np.array([0.0]),
                    lams_2 = np.logspace(start=0, stop=-4, num=10, base=10.0),
                    path = None,
                    logging = False):
    """Hyperparameter grid search over smoothness penalty for nonparametric additive models with interactions
    
    Hyperparameter grid search over smoothness penalty, for each smoothness penalty L0path is run with warm-starts, 
    active set updates and cached matrix factorizations.
    Args:
        L0path_HS: function for grid search along L0path, callable.
        MIP_HS: function for optimizing the MIP with strong hierarchy, callable.
            relies on gurobi.
        beta: coefficients for main effects, list of arrays of shapes [(Ki+1, 1), ...]
        zeta: binary vector to track which main effects are in the active set, bool array of shape [(1, d)]
        delta: coefficients for interaction effects, list of arrays of shapes [(Kij+1, 1), ...]
        alpha: binary vector to track which interaction effects are in the active set, bool array of shape [(1, Imax)]        
        B: B-spline transformed sparse matrices for main effects, list of sparse matrices of shapes [(N, Ki+1), ...].
        B_interaction: B-spline transformed sparse matrices for interaction effects, list of sparse matrices of shapes [(N, Kij+1), ...].
        column_names: names of covariates, array of str.
        lams_1: smoothness penalty for b-splines, array of float scalers.
        lams_2: L0 penalty for b-splines, array of float scalers.
        path: folder path to log results to, str.
        logging: whether to log results to a file, bool scaler.
    
    Returns:
        optimal_solution: (beta_opt, delta_opt, zeta_opt, alpha_opt, lam_1_opt, lam_2_opt, tau_opt, J_opt, active_set_opt, active_interaction_set_opt).
        sparse_solution: (beta_sp, delta_sp, zeta_sp, alpha_sp, lam_1_sp, lam_2_sp, tau_sp, J_sp, active_set_sp, active_interaction_set_sp).
    """
    
    d = len(B)
    N, _ = B[0].shape
        
    val_loss_opt = np.inf
    val_loss_sp = np.inf
    sparsity_opt = d+len(interaction_terms)
    beta = [deepcopy(beta)]*lams_2.shape[0]
    zeta = [deepcopy(zeta)]*lams_2.shape[0]
    delta = [deepcopy(delta)]*lams_2.shape[0]
    alpha = [deepcopy(alpha)]*lams_2.shape[0]
    if logging==True:
        with open(path+'/Training-HS.csv', "w") as f:
            f.write('lambda_1,lambda_2,tau,train,val,Obj,Main-Effects,Interaction-Effects\n') 
    print('lambda_1,lambda_2,tau,train,val,Obj,Main-Effects,Interaction-Effects')
    df = pd.DataFrame(columns=['lam_sm', 'lam_L0', 'tau', *column_names])
    with open(os.path.join(path, 'main_support_regularization_path.csv'), 'w') as f:
        df.to_csv(f, header=True, index=False)
    df = pd.DataFrame(columns=['lam_sm', 'lam_L0', 'tau', *[(column_names[i], column_names[j]) for i, j in interaction_terms]])
    with open(os.path.join(path, 'interaction_support_regularization_path.csv'), 'w') as f:
        df.to_csv(f, header=True, index=False)

    for i, lam_1 in tqdm_notebook(enumerate(lams_1), desc='$\lambda_1$'):
        parameters_path, optimal_solution_path, sparse_solution_path = L0path_HS(MIP_HS = MIP_HS, 
                                                                                 lam_1 = deepcopy(lam_1),
                                                                                 lams_2 = deepcopy(lams_2),
                                                                                 beta = deepcopy(beta),
                                                                                 zeta = deepcopy(zeta),
                                                                                 delta = deepcopy(delta),
                                                                                 alpha = deepcopy(alpha),
                                                                                 start = i)
        beta, delta, zeta, alpha = parameters_path
        beta_opt_path, delta_opt_path, zeta_opt_path, alpha_opt_path, lam_1_opt_path, lam_2_opt_path, tau_opt_path, J_opt_path, active_set_opt_path, active_interaction_set_opt_path, val_loss_opt_path = optimal_solution_path
        beta_sp_path, delta_sp_path, zeta_sp_path, alpha_sp_path, lam_1_sp_path, lam_2_sp_path, tau_sp_path, J_sp_path, active_set_sp_path, active_interaction_set_sp_path, val_loss_sp_path = sparse_solution_path
        if val_loss_opt_path <  val_loss_opt:
            val_loss_opt = deepcopy(val_loss_opt_path)
            beta_opt = deepcopy(beta_opt_path) 
            zeta_opt = deepcopy(zeta_opt_path) 
            delta_opt = deepcopy(delta_opt_path) 
            alpha_opt = deepcopy(alpha_opt_path)
            active_set_opt = deepcopy(active_set_opt_path)
            active_interaction_set_opt = deepcopy(active_interaction_set_opt_path)
            tau_opt = deepcopy(tau_opt_path)
            lam_1_opt = deepcopy(lam_1_opt_path)
            lam_2_opt = deepcopy(lam_2_opt_path)
            J_opt = deepcopy(J_opt_path)
            val_loss_sp = deepcopy(val_loss_sp_path)
            beta_sp = deepcopy(beta_sp_path) 
            zeta_sp = deepcopy(zeta_sp_path) 
            delta_sp = deepcopy(delta_sp_path) 
            alpha_sp = deepcopy(alpha_sp_path)
            active_set_sp = deepcopy(active_set_sp_path)
            active_interaction_set_sp = deepcopy(active_interaction_set_sp_path)
            tau_sp = deepcopy(tau_sp_path)
            lam_1_sp = deepcopy(lam_1_sp_path)
            lam_2_sp = deepcopy(lam_2_sp_path)
            J_sp = deepcopy(J_sp_path)
            
    if logging==True:               
        with open(path+'/Results-HS.txt', "a") as f:
            f.write('Optimal: \lambda_1: {:.7f},\lambda_2: {:.7f}, tau: {:.7f}, val: {:.6f}, J: {:.6f},Main-Effects: {},Interaction Effects: {}\n'.format(lam_1_opt,lam_2_opt,tau_opt, val_loss_opt,J_opt,np.count_nonzero(zeta_opt[0,:]),np.count_nonzero(alpha_opt[0,:]))) 
            f.write('Sparse: \lambda_1: {:.7f},\lambda_2: {:.7f}, tau: {:.7f}, val: {:.6f}, J: {:.6f},Main-Effects: {},Interaction Effects: {}\n'.format(lam_1_sp,lam_2_sp,tau_sp,val_loss_sp,J_opt,np.count_nonzero(zeta_sp[0,:]),np.count_nonzero(alpha_sp[0,:]))) 
#     display(Math(r'Optimal~~~\lambda_1: {:.6f}, \lambda_2: {:.6f}, Val-MAE: {:.6f}, '.format(lam_1_opt,lam_2_opt,val_loss_opt) + '\sum_{j \in S^c} z_j:'+'{}, '.format(np.count_nonzero(zeta_opt[0,:])) + '\sum_{ij \in S^c}z_{ij}:' + '{}.'.format(np.count_nonzero(alpha_opt[0,:]))))

        beta_opt_save = np.empty(len(beta_opt), np.object)
        beta_opt_save[:] = beta_opt
        delta_opt_save = np.empty(len(delta_opt), np.object)
        delta_opt_save[:] = delta_opt
        beta_sp_save = np.empty(len(beta_sp), np.object)
        beta_sp_save[:] = beta_sp
        delta_sp_save = np.empty(len(delta_sp), np.object)
        delta_sp_save[:] = delta_sp

        np.savez_compressed(path+'/optimal_solution_HS',
                            lam_1=lam_1_opt,
                            lam_2=lam_2_opt,
                            tau=tau_opt,
                            beta = beta_opt_save,
                            zeta = zeta_opt,
                            active_set = active_set_opt,
                            active_set_names = [column_names[k] for k in active_set_opt],
                            delta = delta_opt_save,
                            alpha = alpha_opt,
                            active_interaction_set = [interaction_terms[k] for k in active_interaction_set_opt],  
                            active_interaction_set_indices = active_interaction_set_opt,
                            active_interaction_set_names = [(column_names[i], column_names[j]) for i, j in [interaction_terms[k] for k in active_interaction_set_opt]])
        np.savez_compressed(path+'/sparse_solution_HS',
                            lam_1=lam_1_sp,
                            lam_2=lam_2_sp,
                            tau=tau_sp,
                            beta = beta_sp_save,
                            zeta = zeta_sp,
                            active_set = active_set_sp,
                            active_set_names = [column_names[k] for k in active_set_sp],
                            delta = delta_sp_save,
                            alpha = alpha_sp,
                            active_interaction_set = [interaction_terms[k] for k in active_interaction_set_sp],  
                            active_interaction_set_indices = active_interaction_set_sp,
                            active_interaction_set_names = [(column_names[i], column_names[j]) for i, j in [interaction_terms[k] for k in active_interaction_set_sp]])

    print('Optimal: \lambda_1: {:.7f},\lambda_2: {:.7f}, tau: {:.7f}, val: {:.6f}, J: {:.6f},Main-Effects: {},Interaction Effects: {}\n'.format(lam_1_opt,lam_2_opt,tau_opt,val_loss_opt,J_opt,np.count_nonzero(zeta_opt[0,:]),np.count_nonzero(alpha_opt[0,:])))
    print('Sparse: \lambda_1: {:.7f},\lambda_2: {:.7f}, tau: {:.7f}, val: {:.6f}, J: {:.6f},Main-Effects: {},Interaction Effects: {}\n'.format(lam_1_sp,lam_2_sp,tau_sp,val_loss_sp,J_opt,np.count_nonzero(zeta_sp[0,:]),np.count_nonzero(alpha_sp[0,:])))

    return (beta_opt, delta_opt, zeta_opt, alpha_opt, lam_1_opt, lam_2_opt, tau_opt, J_opt, active_set_opt, active_interaction_set_opt),(beta_sp, delta_sp, zeta_sp, alpha_sp, lam_1_sp, lam_2_sp, tau_sp, J_sp, active_set_sp, active_interaction_set_sp)

