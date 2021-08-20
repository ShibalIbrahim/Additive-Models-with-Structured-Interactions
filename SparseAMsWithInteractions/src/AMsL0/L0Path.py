"""Hyperparameter grid search for L0 penalty for nonparametric additive models"""
from copy import deepcopy
from IPython.display import Math
from ipywidgets import *
import numpy as np
import pandas as pd
from patsy import dmatrix
import scipy.sparse as sp
from scipy.special import comb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tnrange, notebook
import warnings

def L0Path(CD_S=None, 
           CD_S_AS=None,
           lam_1=None,
           lams_2=None,
           active_set=None,
           beta=None,
           zeta=None,
           B = None,
           BT_B = None,
           K_main = None, 
           Xval = None,
           Xmin = None,
           Xmax = None,
           Y = None,
           Yval = None,
           y_scaler = None,
           S = None,
           main_terms=None,
           eval_criteria = None,
           path = None,
           logging = False,
           terminate_val_L0path=True):
    """Hyperparameter grid search for L0 penalty for nonparametric additive models
    
    Args:
        CD_S: function for cyclic block coordinate descent, callable.
        CD_S_AS: function for cyclic block coordinate descent over an active set, callable.
        lam_1: smoothness penalty for b-splines, float scaler.
        lams_2: L0 penalty for b-splines, array of float scalers.
        active_set: indices of main effects to optimize over, a numpy int array.
        beta: coefficients for main effects, list of arrays of shapes [(Ki+1, 1), ...].
        zeta: binary vector to track which main effects are in the active set, a bool array of shape (1, d)
        B: B-spline transformed sparse matrices for main effects, list of sparse matrices of shapes [(N, Ki+1), ...].
        BT_B: B^T*B matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
        K_main: Number of knots used for each main effect, a list of int scalers of shape (d,) 
        Xval: validation covariates, a float numpy array of shape (Nval, p).
        Xmin: minimum values of X for all covariates, needed for spline generation, a float numpy array of shape (1, d).
        Xmax: maximum values of X for all covariates, needed for spline generation, a float numpy array of shape (1, d).
        Y: training target responses, a float numpy array of shape (N,).
        Yval: validation target responses, a float numpy array of shape (Nval,).
        y_scaler: sklearn transformation object on responses to inverse transform the responses, see data_utils.py
            supports z-normalization/identity.
        S: Smoothness matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
        main_terms: list of main effects to consider if only a subset need to be considered, 
            not supported yet.
        eval_criteria: evaluation metric for hyperparameter tuning,
          - 'mse', 'mae'
        path: folder path to log results to, str.
        logging: whether to log results to a file, bool scaler.
        terminate_val_L0path: whether to terminate L0 path when the validation performance increases by 1% along the grid, bool.
        
    Returns:
        parameters_path: (beta, zeta, active_set),
        optimal_solution_path: (beta_opt, zeta_opt, lam_1_opt, lam_2_opt, J_opt, active_set_opt, val_loss_opt),
        sparse_solution_path: (beta_sp, zeta_sp, lam_1_sp, lam_2_sp, J_sp, active_set_sp, val_loss_sp),
        union_set_path:
        solution_path: (beta_path, zeta_path)
        
    """
    d = len(B)
    N, _ = B[0].shape
    val_loss_opt = np.inf
    val_loss = np.inf*np.ones((lams_2.shape[0],),dtype=float)
    val_std_err = np.inf*np.ones((lams_2.shape[0],),dtype=float)
    beta_path = []
    zeta_path = []
    sparsity = (d)*np.ones((lams_2.shape[0],),dtype=float)
    J = np.zeros((lams_2.shape[0],),dtype=float)
    MaxSuppSize_flag = 0
    warnings.filterwarnings("error")
    P = []
    eps = 1e-8
    if eval_criteria == 'mse':
        evaluate = mean_squared_error
    elif eval_criteria == 'mae':
        evaluate = mean_absolute_error
    else:
        raise ValueError("Evaluation criteria {} is not supported".format(eval_criteria))
    
    # Cached matrix factorizations
    P = [sp.linalg.splu(BkT_Bk+2*N*(lam_1*Sk+eps*sp.csr_matrix(np.identity(Bk.shape[1])))) for Bk, BkT_Bk, Sk in zip(B, BT_B, S)]
#     for Bk, BkT_Bk, Sk in zip(B, BT_B, S):
#         try:
#             p = sp.linalg.splu(BkT_Bk+2*N*lam_1*Sk)
#         except:
#             p = sp.linalg.splu(BkT_Bk+2*N*(lam_1*Sk+eps*sp.csr_matrix(np.identity(Bk.shape[1]))))
#         P.append(p)

    active_union_set = []
    
    # Search over L0 path (lambda_2)
    for j, lam_2 in enumerate(lams_2):
        
        if active_set[j] is None:            
            if j==0:
                active_set_current = deepcopy(active_set[0])
                beta_current = deepcopy(beta[0])
                zeta_current = deepcopy(zeta[0])
            else:
                active_set_current = deepcopy(active_set[j-1])
                beta_current = deepcopy(beta[j-1])
                zeta_current = deepcopy(zeta[j-1])
        else:
            active_set_current = deepcopy(active_set[j])
            beta_current = deepcopy(beta[j])
            zeta_current = deepcopy(zeta[j])

        if active_set[j] is None and j==0:
            Ypred = np.mean(Y)*np.ones(Y.shape,dtype=float)
        else:        
            if len(active_set_current)==0 :
                Ypred = np.mean(Y)*np.ones(Y.shape,dtype=float)
            elif len(active_set_current)>0 :
                Ypred = np.mean(Y)*np.ones(Y.shape,dtype=float)\
                        +np.array(sum([(B[k]).dot(beta_current[k]) for k in active_set_current])).reshape(Y.shape)

        '''
            CD on main effects
        '''
        Ypred, beta[j], zeta[j], active_set[j], MaxSuppSize_flag = CD_S(CD_S_AS=CD_S_AS, Ypred=Ypred, B=B, BT_B=BT_B, S=S, P=P, I=len(main_terms), beta=deepcopy(beta_current), zeta=deepcopy(zeta_current), lam=[lam_1, lam_2], active_set=deepcopy(active_set_current), full_set=main_terms)
        
        if MaxSuppSize_flag==1:
            break

        train_loss = evaluate(y_scaler.inverse_transform(Y), y_scaler.inverse_transform(Ypred))
        
        # Generate b-splines for validation set for active set
        Bval = [None]*d
        for k in active_set[j]:
            Bval[k] = sp.csr_matrix(np.array(dmatrix("bs(x, df={}, degree=3, include_intercept=False, lower_bound={}, upper_bound={})".format(K_main[k], Xmin[k], Xmax[k]), {"x": Xval[:,k]})),dtype=np.float64)
                    
        if len(active_set[j])==0:
            Yvalpred = np.mean(Y)*np.ones(Yval.shape, dtype=float)
        elif len(active_set[j])>0:
            Yvalpred = np.mean(Y)*np.ones(Yval.shape,dtype=float)\
                       +np.array(sum([(Bval[k]).dot(beta[j][k]) for k in active_set[j]])).reshape(Yval.shape)
        val_loss[j] = evaluate(y_scaler.inverse_transform(Yval), y_scaler.inverse_transform(Yvalpred))
        val_std_err[j] = (mean_squared_error(y_scaler.inverse_transform(Yval), y_scaler.inverse_transform(Yvalpred))**0.5)/(Yval.shape[0]**0.5)
        sparsity[j] = np.count_nonzero(zeta[j][0,:])
        J[j] = 0.5*mean_squared_error(Y, Ypred)+\
               lam_1*sum([(np.transpose(beta[j][k])).dot(S[k].dot(beta[j][k]))[0,0] for k in active_set[j]])+\
               eps*sum([np.dot(beta[j][k][:,0],beta[j][k][:,0]) for k in active_set[j]])+\
               lam_2*(np.count_nonzero(zeta[j][0,:]))
        if logging ==True:
            with open(os.path.join(path, 'Training.csv'), "a") as f:
                f.write('{0:.8f},   {1:.8f},   {2:2.6f},   {3:2.6f},   {4:2.6f},   {5:4d}\n'.format(lam_1,lam_2,train_loss, val_loss[j], J[j],np.count_nonzero(zeta[j][0,:]))) 
        print('{0:.8f},   {1:.8f},   {2:2.6f},   {3:2.6f},   {4:2.6f},   {5:4d}'.format(lam_1,lam_2,train_loss, val_loss[j], J[j],np.count_nonzero(zeta[j][0,:])))
        df = pd.DataFrame(columns=[lam_1, lam_2, *(zeta[j][0,:])])
        with open(os.path.join(path, 'support_regularization_path.csv'), 'a') as f:
            df.to_csv(f, header=True, index=False)
        beta_path.append(deepcopy(beta[j]))
        zeta_path.append(deepcopy(zeta[j]))
        if val_loss[j] <  val_loss_opt:
            val_loss_opt = deepcopy(val_loss[j])
            val_std_err_opt = deepcopy(val_std_err[j])
            beta_opt = deepcopy(beta[j]) 
            zeta_opt = deepcopy(zeta[j]) 
            active_set_opt = deepcopy(active_set[j])
            lam_1_opt = deepcopy(lam_1)
            lam_2_opt = deepcopy(lam_2)
            J_opt = deepcopy(J[j])

        if (val_loss[j]!=np.inf) and (val_loss_opt!=np.inf) and (val_loss[j]>1.01*val_loss_opt) and terminate_val_L0path:
            MaxSuppSize_flag==1
            break
        else:
            active_union_set =  sorted(list(set(active_union_set) | set(active_set[j])))
            
#     val_loss_percent = ((val_loss-val_loss_opt*np.ones((lams_2.shape[0],),dtype=float))/(val_loss_opt*np.ones((lams_2.shape[0],),dtype=float)))*100
    if eval_criteria == 'mse':
        val_loss_diff = val_loss**0.5 - val_loss_opt**0.5
    elif eval_criteria == 'mae':
        val_loss_diff = val_loss - val_loss_opt
    else:
        raise ValueError("Evaluation criteria {} is not supported".format(eval_criteria))
#     subset_indices = np.where(val_loss_percent < 1)[0]                         
    subset_indices = np.where(val_loss_diff < val_std_err_opt)[0]                         
    sparsity_subset = sparsity[subset_indices]  
    min_sparsity_subset_indices = subset_indices[np.argwhere(sparsity_subset == np.amin(sparsity_subset))].reshape(-1,)
    min_sparsity_min_val_index = min_sparsity_subset_indices[np.argwhere(val_loss[min_sparsity_subset_indices] == np.amin(val_loss[min_sparsity_subset_indices])).reshape(-1,)][0]
                         
    val_loss_sp = deepcopy(val_loss[min_sparsity_min_val_index])
    beta_sp = deepcopy(beta[min_sparsity_min_val_index]) 
    zeta_sp = deepcopy(zeta[min_sparsity_min_val_index]) 
    active_set_sp = deepcopy(active_set[min_sparsity_min_val_index])
    lam_1_sp = deepcopy(lam_1)
    lam_2_sp = deepcopy(lams_2[min_sparsity_min_val_index])
    J_sp = deepcopy(J[min_sparsity_min_val_index])
    
    return (beta, zeta, active_set), (beta_opt, zeta_opt, lam_1_opt, lam_2_opt, J_opt, active_set_opt, val_loss_opt), (beta_sp, zeta_sp, lam_1_sp, lam_2_sp, J_sp, active_set_sp, val_loss_sp), active_union_set, (beta_path, zeta_path)