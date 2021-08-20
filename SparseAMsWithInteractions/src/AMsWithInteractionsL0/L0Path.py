from copy import deepcopy
from IPython.display import Math
from ipywidgets import *
import numpy as np
import pandas as pd
from patsy import dmatrix
import scipy.sparse as sp
from scipy.special import comb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import notebook
import warnings

def L0Path(CD_J = None, 
           CD_J_AS = None, 
           lam_1 = None,
           lams_2 = None,
           active_set = None,
           active_interaction_set = None,
           beta = None,
           zeta = None,
           delta = None,
           alpha = None, 
           B = None,
           BT_B = None,
           B_interaction = None,
           B_interactionT_B_interaction = None,
           K_main = None, 
           K_interaction = None, 
           Xval = None,
           Xmin = None,
           Xmax = None,
           Y = None,
           Yval = None,
           y_scaler = None,
           S = None,
           S_interaction = None,
           interaction_terms = None,
           eval_criteria = None,
           path = None,
           r = None,
           logging = False,
           terminate_val_L0path=True):
    """Hyperparameter grid search for L0 penalty for nonparametric additive models with interactions
    
    Args:
        CD_J: function for cyclic block coordinate descent, callable.
        CD_J_AS: function for cyclic block coordinate descent over an active set, callable.
        lam_1: smoothness penalty for b-splines, float scaler.
        lams_2: L0 penalty for b-splines, array of float scalers.
        active_set: indices of main effects to optimize over, a numpy int array.
        active_interaction_set: indices of interaction effects to optimize over, a numpy int array.
        beta: coefficients for main effects, list of arrays of shapes [(Ki+1, 1), ...].
        zeta: binary vector to track which main effects are in the active set, a bool array of shape (1, d)
            corresponds to z_i's in the paper.
        delta: coefficients for interaction effects, list of arrays of shapes [(Kij+1, 1), ...].
            corresponds to theta in the paper.
        alpha: binary vector to track which interactions effects are in the active interaction set, a bool array of shape (1, Imax)
            corresponds to z_ij's in the paper.
        B: B-spline transformed sparse matrices for main effects, list of sparse matrices of shapes [(N, Ki+1), ...].
        BT_B: B^T*B matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
        B_interaction: B-spline transformed sparse matrices for interaction effects, list of sparse matrices of shapes [(N, Kij+1), ...].
        B_interactionT_B_interaction: B^T*B matrices for interaction effects, list of sparse matrices of shapes [(Kij+1, Kij+1), ...].
        K_main: Number of knots used for each main effect, a list of int scalers of shape (d,) 
        K_interaction: Number of knots used for each interaction effect, a list of int scalers of shape (Imax,) 
        Xval: validation covariates, a float numpy array of shape (Nval, p).
        Xmin: minimum values of X for all covariates, needed for spline generation, a float numpy array of shape (1, d).
        Xmax: maximum values of X for all covariates, needed for spline generation, a float numpy array of shape (1, d).
        Y: training target responses, a float numpy array of shape (N,).
        Yval: validation target responses, a float numpy array of shape (Nval,).
        y_scaler: sklearn transformation object on responses to inverse transform the responses, see data_utils.py
            supports z-normalization/identity.
        S: Smoothness matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
        S_interaction: Smoothness matrices for interaction effects, list of sparse matrices of shapes [(Kij+1, Kij+1), ...].
        interaction_terms: list of interaction effects to consider if only a subset need to be considered, 
            a 2D numpy array of of shape (Imax, 2).
        eval_criteria: evaluation metric for hyperparameter tuning,
          - 'mse', 'mae'
        path: folder path to log results to, str.
        r: relative scaling factor for L0 penalty between main and interaction effects.
            We consider r=1.0 (corresponds to alpha symbol in the paper), float scaler. 
        logging: whether to log results to a file, bool scaler.
        terminate_val_L0path: whether to terminate L0 path when the validation performance increases by 1% along the grid, bool.
        
    Returns:
        parameters_path: (beta, delta, zeta, alpha, active_set, active_interaction_set).
        optimal_solution_path: (beta_opt, delta_opt, zeta_opt, alpha_opt, lam_1_opt, lam_2_opt, J_opt, active_set_opt, active_interaction_set_opt, val_loss_opt).
        sparse_solution_path: (beta_sp, delta_sp, zeta_sp, alpha_sp, lam_1_sp, lam_2_sp, J_sp, active_set_sp, active_interaction_set_sp, val_loss_sp).
        union_set_path: (active_union_set, active_interaction_union_set).        
    """
    
    d = len(B)
    N, _ = B[0].shape
    val_loss_opt = np.inf
    val_loss = np.inf*np.ones((lams_2.shape[0],),dtype=float)
    val_std_err = np.inf*np.ones((lams_2.shape[0],),dtype=float)
    sparsity = (d+np.floor(comb(d, 2, exact=False)))*np.ones((lams_2.shape[0],),dtype=float)
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
    P = [sp.linalg.splu(BkT_Bk+2*N*(lam_1*Sk+eps*sp.csr_matrix(np.identity(Bk.shape[1])))) for Bk, BkT_Bk, Sk in zip(B, BT_B, S)]
    P_interaction = [sp.linalg.splu(BkT_Bk+2*N*(lam_1*Sk+eps*sp.csr_matrix(np.identity(Bk.shape[1])))) for Bk, BkT_Bk, Sk in zip(B_interaction, B_interactionT_B_interaction, S_interaction)]

    active_union_set = []
    active_interaction_union_set = []
    
    for j, lam_2 in enumerate(lams_2):
        
        # Warm-starting solutions and active sets
        if active_set[j] is None:            
            if j==0:
                active_set_current = deepcopy(active_set[0])
                active_interaction_set_current = deepcopy(active_interaction_set[0])
                beta_current = deepcopy(beta[0])
                zeta_current = deepcopy(zeta[0])
                delta_current = deepcopy(delta[0])
                alpha_current = deepcopy(alpha[0])
            else:
                active_set_current = deepcopy(active_set[j-1])
                active_interaction_set_current = deepcopy(active_interaction_set[j-1])
                beta_current = deepcopy(beta[j-1])
                zeta_current = deepcopy(zeta[j-1])
                alpha_current = deepcopy(alpha[j-1])
                delta_current = deepcopy(delta[j-1])
        else:
            active_set_current = deepcopy(active_set[j])
            active_interaction_set_current = deepcopy(active_interaction_set[j])
            beta_current = deepcopy(beta[j])
            zeta_current = deepcopy(zeta[j])
            alpha_current = deepcopy(alpha[j])
            delta_current = deepcopy(delta[j])

        if active_set[j] is None and j==0:
            Ypred = np.mean(Y)*np.ones(Y.shape,dtype=float)
        else:        
            if len(active_set_current)==0 and len(active_interaction_set_current)==0:
                Ypred = np.mean(Y)*np.ones(Y.shape,dtype=float)
            elif len(active_set_current)==0 and len(active_interaction_set_current)>0:
                Ypred = np.mean(Y)*np.ones(Y.shape,dtype=float)\
                        +np.array(sum([(B_interaction[k]).dot(delta_current[k]) for k in active_interaction_set_current])).reshape(Y.shape)
            elif len(active_set_current)>0 and len(active_interaction_set_current)==0:
                Ypred = np.mean(Y)*np.ones(Y.shape,dtype=float)\
                        +np.array(sum([(B[k]).dot(beta_current[k]) for k in active_set_current])).reshape(Y.shape)
            elif len(active_set_current)>0 and len(active_interaction_set_current)>0:
                Ypred = np.mean(Y)*np.ones(Y.shape,dtype=float)\
                        +np.array(sum([(B[k]).dot(beta_current[k]) for k in active_set_current])).reshape(Y.shape)\
                        +np.array(sum([(B_interaction[k]).dot(delta_current[k]) for k in active_interaction_set_current])).reshape(Y.shape)        

        '''
            Joint Optimization
        '''
        Ypred, beta[j], zeta[j], delta[j], alpha[j], active_set[j], active_interaction_set[j], MaxSuppSize_flag = CD_J(CD_J_AS=CD_J_AS, Ypred=Ypred, beta=[deepcopy(beta_current), deepcopy(delta_current)], zeta=[deepcopy(zeta_current), deepcopy(alpha_current)], lam=[lam_1, lam_2], active_set=[deepcopy(active_set_current), deepcopy(active_interaction_set_current)], P=P, P_interaction=P_interaction)
    
                        
        if MaxSuppSize_flag==1:
            break            

        train_loss = evaluate(y_scaler.inverse_transform(Y), y_scaler.inverse_transform(Ypred))
        
        # Generate b-splines for validation set for active set
        Bval = [None]*d
        for k in active_set[j]:
            Bval[k] = sp.csr_matrix(np.array(dmatrix("bs(x, df={}, degree=3, include_intercept=False, lower_bound={}, upper_bound={})".format(K_main[k], Xmin[k], Xmax[k]), {"x": Xval[:,k]})),dtype=np.float64)
        Bval_interaction = [None]*len(interaction_terms)
        for k in active_interaction_set[j]:
            f_i, f_j = interaction_terms[k]
            Bval_interaction[k] = sp.csr_matrix(np.array(dmatrix("te(bs(x1, df={}, degree=3, include_intercept=False, lower_bound={}, upper_bound={}), bs(x2, df={}, degree=3, include_intercept=False, lower_bound={}, upper_bound={}))".format(K_interaction[f_i], Xmin[f_i], Xmax[f_i], K_interaction[f_j], Xmin[f_j], Xmax[f_j]), {"x1": Xval[:,f_i], "x2": Xval[:,f_j]})),dtype=np.float64)
                    
        if len(active_set[j])==0 and len(active_interaction_set[j])==0:
            Yvalpred = np.mean(Y)*np.ones(Yval.shape,dtype=float)
        elif len(active_set[j])==0 and len(active_interaction_set[j])>0:
            Yvalpred = np.mean(Y)*np.ones(Yval.shape,dtype=float)\
                       +np.array(sum([(Bval_interaction[k]).dot(delta[j][k]) for k in active_interaction_set[j]])).reshape(Yval.shape)
        elif len(active_set[j])>0 and len(active_interaction_set[j])==0:
            Yvalpred = np.mean(Y)*np.ones(Yval.shape,dtype=float)\
                       +np.array(sum([(Bval[k]).dot(beta[j][k]) for k in active_set[j]])).reshape(Yval.shape)
        elif len(active_set[j])>0 and len(active_interaction_set[j])>0:
            Yvalpred = np.mean(Y)*np.ones(Yval.shape,dtype=float)\
                       +np.array(sum([(Bval[k]).dot(beta[j][k]) for k in active_set[j]])).reshape(Yval.shape)\
                       +np.array(sum([(Bval_interaction[k]).dot(delta[j][k]) for k in active_interaction_set[j]])).reshape(Yval.shape)        
        val_loss[j] = evaluate(y_scaler.inverse_transform(Yval), y_scaler.inverse_transform(Yvalpred))
        val_std_err[j] = (mean_squared_error(y_scaler.inverse_transform(Yval), y_scaler.inverse_transform(Yvalpred))**0.5)/(Yval.shape[0]**0.5)
        sparsity[j] = np.count_nonzero(zeta[j][0,:]) + np.count_nonzero(alpha[j][0,:])
        J[j] = 0.5*mean_squared_error(Y, Ypred)+\
               lam_1*sum([(np.transpose(beta[j][k])).dot(S[k].dot(beta[j][k]))[0,0] for k in active_set[j]])+\
               lam_1*sum([(np.transpose(delta[j][k])).dot(S_interaction[k].dot(delta[j][k]))[0,0] for k in active_interaction_set[j]])+\
               eps*sum([np.dot(beta[j][k][:,0],beta[j][k][:,0]) for k in active_set[j]])+\
               eps*sum([np.dot(delta[j][k][:,0],delta[j][k][:,0]) for k in active_interaction_set[j]])+\
               lam_2*(np.count_nonzero(zeta[j][0,:]))+\
               r*lam_2*(np.count_nonzero(alpha[j][0,:]))    
        if logging ==True:
            with open(path+'/Training.csv', "a") as f:
                f.write('{0:.8f},   {1:.8f},   {2:2.6f},   {3:2.6f},   {4:2.6f},   {5:4d},   {6:4d}\n'.format(
                    lam_1, lam_2, train_loss, val_loss[j], J[j],np.count_nonzero(zeta[j][0,:]),np.count_nonzero(alpha[j][0,:]))) 
        print('{0:.8f},   {1:.8f},   {2:2.6f},   {3:2.6f},   {4:2.6f},   {5:4d},   {6:4d}'.format(lam_1,lam_2,train_loss, val_loss[j], J[j],np.count_nonzero(zeta[j][0,:]),np.count_nonzero(alpha[j][0,:])))
        df = pd.DataFrame(columns=[lam_1, lam_2, *(zeta[j][0,:])])
        with open(os.path.join(path, 'main_support_regularization_path.csv'), 'a') as f:
            df.to_csv(f, header=True, index=False)
        df = pd.DataFrame(columns=[lam_1, lam_2, *(alpha[j][0,:])])
        with open(os.path.join(path, 'interaction_support_regularization_path.csv'), 'a') as f:
            df.to_csv(f, header=True, index=False)

        if val_loss[j] <  val_loss_opt:
            val_loss_opt = deepcopy(val_loss[j])
            val_std_err_opt = deepcopy(val_std_err[j])
            beta_opt = deepcopy(beta[j]) 
            zeta_opt = deepcopy(zeta[j]) 
            delta_opt = deepcopy(delta[j]) 
            alpha_opt = deepcopy(alpha[j])
            active_set_opt = deepcopy(active_set[j])
            active_interaction_set_opt = deepcopy(active_interaction_set[j])
            lam_1_opt = deepcopy(lam_1)
            lam_2_opt = deepcopy(lam_2)
            J_opt = deepcopy(J[j])

        if (val_loss[j]!=np.inf) and (val_loss_opt!=np.inf) and (val_loss[j]>1.01*val_loss_opt) and terminate_val_L0path:
            MaxSuppSize_flag==1
            break
        else:
            active_union_set =  sorted(list(set(active_union_set) | set(active_set[j])))
            active_interaction_union_set = sorted(list(set(active_interaction_union_set) | set(active_interaction_set[j])))
            
#     val_loss_percent = ((val_loss-val_loss_opt*np.ones((lams_2.shape[0],),dtype=float))/(val_loss_opt*np.ones((lams_2.shape[0],),dtype=float)))*100
    if eval_criteria == 'mse':
        val_loss_diff = val_loss**0.5 - val_loss_opt**0.5
    elif eval_criteria == 'mae':
        val_loss_diff = val_loss - val_loss_opt
    else:
        raise ValueError("Evaluation criteria {} is not supported".format(eval_criteria))
#     subset_indices = np.where(val_loss_percent<1)[0]                         
    subset_indices = np.where(val_loss_diff<val_std_err_opt)[0]                         
    sparsity_subset = sparsity[subset_indices]  
    min_sparsity_subset_indices = subset_indices[np.argwhere(sparsity_subset == np.amin(sparsity_subset))].reshape(-1,)
    min_sparsity_min_val_index = min_sparsity_subset_indices[np.argwhere(val_loss[min_sparsity_subset_indices] == np.amin(val_loss[min_sparsity_subset_indices])).reshape(-1,)][0]
                         
    val_loss_sp = deepcopy(val_loss[min_sparsity_min_val_index])
    beta_sp = deepcopy(beta[min_sparsity_min_val_index]) 
    zeta_sp = deepcopy(zeta[min_sparsity_min_val_index]) 
    delta_sp = deepcopy(delta[min_sparsity_min_val_index]) 
    alpha_sp = deepcopy(alpha[min_sparsity_min_val_index])
    active_set_sp = deepcopy(active_set[min_sparsity_min_val_index])
    active_interaction_set_sp = deepcopy(active_interaction_set[min_sparsity_min_val_index])
    lam_1_sp = deepcopy(lam_1)
    lam_2_sp = deepcopy(lams_2[min_sparsity_min_val_index])
    J_sp = deepcopy(J[min_sparsity_min_val_index])
    
    return (beta, delta, zeta, alpha, active_set, active_interaction_set), (beta_opt, delta_opt, zeta_opt, alpha_opt, lam_1_opt, lam_2_opt, J_opt, active_set_opt, active_interaction_set_opt, val_loss_opt), (beta_sp, delta_sp, zeta_sp, alpha_sp, lam_1_sp, lam_2_sp, J_sp, active_set_sp, active_interaction_set_sp, val_loss_sp), (active_union_set, active_interaction_union_set)