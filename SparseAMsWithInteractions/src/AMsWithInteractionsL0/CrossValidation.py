from copy import deepcopy
from IPython.display import Math
from ipywidgets import *
import numpy as np
import pandas as pd
from scipy.special import comb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import notebook

def CrossValidation(L0path = None,
                    CD_J = None, 
                    CD_J_AS = None, 
                    B = None,
                    B_interaction = None,
                    interaction_terms = None,
                    column_names = None,
                    lams_1 = np.logspace(start=-3, stop=-8, num=20, base=10.0),
                    lams_2 = np.logspace(start=0, stop=-3, num=25, base=10.0),
                    path = None,
                    logging = False):
    """Hyperparameter grid search over smoothness penalty for nonparametric additive models with interactions
    
    Hyperparameter grid search over smoothness penalty, for each smoothness penalty L0path is run with warm-starts, 
    active set updates and cached matrix factorizations.
    Args:
        L0path: function for grid search along L0path, callable.
        CD_J: function for cyclic block coordinate descent, callable.
        CD_J_AS: function for cyclic block coordinate descent over an active set, callable.
        B: B-spline transformed sparse matrices for main effects, list of sparse matrices of shapes [(N, Ki+1), ...].
        B_interaction: B-spline transformed sparse matrices for interaction effects, list of sparse matrices of shapes [(N, Kij+1), ...].
        column_names: names of covariates, array of str.
        lams_1: smoothness penalty for b-splines, array of float scalers.
        lams_2: L0 penalty for b-splines, array of float scalers.
        active_set: indices of main effects to optimize over, a numpy int array.
        path: folder path to log results to, str.
        logging: whether to log results to a file, bool scaler.
    
    Returns:
        optimal_solution: (beta_opt, delta_opt, zeta_opt, alpha_opt, lam_1_opt, lam_2_opt, J_opt, active_set_opt, active_interaction_set_opt).
        sparse_solution: (beta_sp, delta_sp, zeta_sp, alpha_sp, lam_1_sp, lam_2_sp, J_sp, active_set_sp, active_interaction_set_sp).
        union_set_path: (active_union_set, active_interaction_union_set)    
    """
    
    d = len(B)
    N, _ = B[0].shape
        
    val_loss_opt = np.inf
    val_loss_sp = np.inf
    sparsity_opt = d+len(interaction_terms)
    active_set = [None] * lams_2.shape[0]
    active_interaction_set = [None] * lams_2.shape[0]
    beta = [[np.zeros((B[k].shape[1],1),dtype=float) for k in range(d)]] * lams_2.shape[0]
    zeta = [np.zeros((1,d),dtype=int)]* lams_2.shape[0]
    delta = [[np.zeros((B_interaction[k].shape[1],1),dtype=float) for k in range(len(interaction_terms))]]* lams_2.shape[0]
    alpha = [np.zeros((1,len(interaction_terms)),dtype=int)]* lams_2.shape[0]
    if logging==True:
        with open(path+'/Training.csv', "w") as f:
            f.write('{0:8s},        {1:8s},  {2:8s},    {3:8s},    {4:8s},  {5:4s}, {6:4s}\n'.format('Smoothness',  'L0',  'train',   'val',  'obj',   'nnz(M)',   'nnz(I)')) 
#    print('\lambda_1,\lambda_2,Train-MAE,Val-MAE,Obj,Main-Effects,Interaction-Effects')
    print('{0:8s},        {1:8s},  {2:8s},    {3:8s},    {4:8s},  {5:4s}, {6:4s}'.format('Smoothness',  'L0',  'train',   'val',  'obj',   'nnz(M)',   'nnz(I)'))
    df = pd.DataFrame(columns=['lam_sm', 'lam_L0', *column_names])
    with open(os.path.join(path, 'main_support_regularization_path.csv'), 'w') as f:
        df.to_csv(f, header=True, index=False)
    df = pd.DataFrame(columns=['lam_sm', 'lam_L0', *[(column_names[i], column_names[j]) for i, j in interaction_terms]])
    with open(os.path.join(path, 'interaction_support_regularization_path.csv'), 'w') as f:
        df.to_csv(f, header=True, index=False)

    active_union_set = []
    active_interaction_union_set = []
    
    for i, lam_1 in enumerate(lams_1):
        parameters_path, optimal_solution_path, sparse_solution_path, union_set_path = L0path(CD_J = CD_J, 
                                                            CD_J_AS = CD_J_AS, 
                                                            lam_1 = deepcopy(lam_1),
                                                            lams_2 = deepcopy(lams_2),
                                                            active_set = deepcopy(active_set),
                                                            active_interaction_set = deepcopy(active_interaction_set),
                                                            beta = deepcopy(beta),
                                                            zeta = deepcopy(zeta),
                                                            delta = deepcopy(delta),
                                                            alpha = deepcopy(alpha))
        beta, delta, zeta, alpha, active_set, active_interaction_set = parameters_path
        beta_opt_path, delta_opt_path, zeta_opt_path, alpha_opt_path, lam_1_opt_path, lam_2_opt_path, J_opt_path, active_set_opt_path, active_interaction_set_opt_path, val_loss_opt_path = optimal_solution_path
        beta_sp_path, delta_sp_path, zeta_sp_path, alpha_sp_path, lam_1_sp_path, lam_2_sp_path, J_sp_path, active_set_sp_path, active_interaction_set_sp_path, val_loss_sp_path = sparse_solution_path
        

        if val_loss_opt_path <  val_loss_opt:
            val_loss_opt = deepcopy(val_loss_opt_path)
            beta_opt = deepcopy(beta_opt_path) 
            zeta_opt = deepcopy(zeta_opt_path) 
            delta_opt = deepcopy(delta_opt_path) 
            alpha_opt = deepcopy(alpha_opt_path)
            active_set_opt = deepcopy(active_set_opt_path)
            active_interaction_set_opt = deepcopy(active_interaction_set_opt_path)
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
            lam_1_sp = deepcopy(lam_1_sp_path)
            lam_2_sp = deepcopy(lam_2_sp_path)
            J_sp = deepcopy(J_sp_path)
#         if (val_loss_opt_path!=np.inf) and (val_loss_opt!=np.inf) and (val_loss_opt_path>1.01*val_loss_opt):
#             break
#         else:
        active_union_set_path, active_interaction_union_set_path = union_set_path
        active_union_set =  sorted(list(set(active_union_set) | set(active_union_set_path)))
        active_interaction_union_set = sorted(list(set(active_interaction_union_set) | set(active_interaction_union_set_path)))

            
    if logging==True:               
        with open(path+'/Results.txt', "a") as f:
            f.write('Optimal: \lambda_1: {:.7f},\lambda_2: {:.7f}, Val-MAE: {:.6f}, J: {:.6f},Main-Effects: {},Interaction Effects: {}\n'.format(lam_1_opt,lam_2_opt,val_loss_opt,J_opt,np.count_nonzero(zeta_opt[0,:]),np.count_nonzero(alpha_opt[0,:]))) 
            f.write('Sparse: \lambda_1: {:.7f},\lambda_2: {:.7f}, Val-MAE: {:.6f}, J: {:.6f},Main-Effects: {},Interaction Effects: {}\n'.format(lam_1_sp,lam_2_sp,val_loss_sp,J_opt,np.count_nonzero(zeta_sp[0,:]),np.count_nonzero(alpha_sp[0,:]))) 
#     display(Math(r'Optimal~~~\lambda_1: {:.6f}, \lambda_2: {:.6f}, Val-MAE: {:.6f}, '.format(lam_1_opt,lam_2_opt,val_loss_opt) + '\sum_{j \in S^c} z_j:'+'{}, '.format(np.count_nonzero(zeta_opt[0,:])) + '\sum_{ij \in S^c}z_{ij}:' + '{}.'.format(np.count_nonzero(alpha_opt[0,:]))))
        beta_opt_save = np.empty(len(beta_opt), np.object)
        beta_opt_save[:] = beta_opt
        delta_opt_save = np.empty(len(delta_opt), np.object)
        delta_opt_save[:] = delta_opt
        beta_sp_save = np.empty(len(beta_sp), np.object)
        beta_sp_save[:] = beta_sp
        delta_sp_save = np.empty(len(delta_sp), np.object)
        delta_sp_save[:] = delta_sp
        
        np.savez_compressed(path+'/optimal_solution',
                            lam_1=lam_1_opt,
                            lam_2=lam_2_opt,
                            beta=beta_opt_save,
                            zeta=zeta_opt,
                            active_set=active_set_opt,
                            active_set_names=[column_names[k] for k in active_set_opt],
                            delta=delta_opt_save,
                            alpha=alpha_opt,
                            active_interaction_set=[interaction_terms[k] for k in active_interaction_set_opt],  
                            active_interaction_set_indices=active_interaction_set_opt,
                            active_interaction_set_names=[(column_names[i], column_names[j]) for i, j in [interaction_terms[k] for k in active_interaction_set_opt]])
        np.savez_compressed(path+'/sparse_solution',
                            lam_1=lam_1_sp,
                            lam_2=lam_2_sp,
                            beta=beta_sp_save,
                            zeta=zeta_sp,
                            active_set=active_set_sp,
                            active_set_names=[column_names[k] for k in active_set_sp],
                            delta=delta_sp_save,
                            alpha=alpha_sp,
                            active_interaction_set=[interaction_terms[k] for k in active_interaction_set_sp],
                            active_interaction_set_indices=active_interaction_set_sp,
                            active_interaction_set_names=[(column_names[i], column_names[j]) for i, j in [interaction_terms[k] for k in active_interaction_set_sp]])
        np.savez_compressed(path+'/interaction_terms',
                            interaction_terms=interaction_terms)
        np.savez_compressed(path+'/union_set',
                            active_union_set=active_union_set,
                            active_union_set_names=[column_names[k] for k in active_union_set],
                            active_interaction_union_set=[interaction_terms[k] for k in active_interaction_union_set],
                            active_interaction_union_set_indices=active_interaction_union_set,
                            active_interaction_union_set_names=[(column_names[i], column_names[j]) for i, j in [interaction_terms[k] for k in active_interaction_union_set]])

    print('Optimal: \lambda_1: {:.7f},\lambda_2: {:.7f}, Val-MAE: {:.6f}, J: {:.6f},Main-Effects: {},Interaction Effects: {}\n'.format(lam_1_opt,lam_2_opt,val_loss_opt,J_opt,np.count_nonzero(zeta_opt[0,:]),np.count_nonzero(alpha_opt[0,:])))
    print('Sparse: \lambda_1: {:.7f},\lambda_2: {:.7f}, Val-MAE: {:.6f}, J: {:.6f},Main-Effects: {},Interaction Effects: {}\n'.format(lam_1_sp,lam_2_sp,val_loss_sp,J_opt,np.count_nonzero(zeta_sp[0,:]),np.count_nonzero(alpha_sp[0,:])))

    return (beta_opt, delta_opt, zeta_opt, alpha_opt, lam_1_opt, lam_2_opt, J_opt, active_set_opt, active_interaction_set_opt),(beta_sp, delta_sp, zeta_sp, alpha_sp, lam_1_sp, lam_2_sp, J_sp, active_set_sp, active_interaction_set_sp), (active_union_set, active_interaction_union_set)

