from copy import deepcopy
from IPython.display import Math
from ipywidgets import *
import numpy as np
import pandas as pd
from patsy import dmatrix
import scipy.sparse as sp
from scipy.special import comb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
from tqdm import tnrange, tqdm_notebook
import warnings

def TauPath(lam_1 = None,
            lam_2 = None,
            beta = None,
            zeta = None,
            delta = None,
            alpha = None,
            P = None,
            P_interaction = None,
            taus = np.logspace(start=0, stop=-2, num=50, base=10),
            CD_J_AS = None,
            active_set = None,
            active_interaction_set = None,
            B = None,
            B_interaction = None,
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
            logging = False):
    """Hyperparameter grid search for tau penalty for nonparametric additive models with interactions under hierarchy
    
    Args:
        lam_1: smoothness penalty for b-splines, float scaler.
        lam_2: L0 penalty for b-splines, float scaler.
        beta: coefficients for main effects, list of arrays of shapes [(Ki+1, 1), ...].
        zeta: binary vector to track which main effects are in the active set, a bool array of shape (1, d)
            corresponds to z_i's in the paper.
        delta: coefficients for interaction effects, list of arrays of shapes [(Kij+1, 1), ...].
            corresponds to theta in the paper.
        alpha: binary vector to track which interactions effects are in the active interaction set, a bool array of shape (1, Imax)
            corresponds to z_ij's in the paper.
        P: B^T*B + 2*N*(lam_1*S_i + eps*I) matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
            eps is a small epsilon for numerical stability.
        P_interaction: B^T*B + 2*N*(lam_1*S_ij + eps*I) matrices for interaction effects, list of sparse matrices of shapes [(Kij+1, Kij+1), ...].
            eps is a small epsilon for numerical stability.
        taus: thresholding penalty for generating feasible subsets of main/interaction effects that maintain strong hierarchy, array of float scalers.
        CD_J_AS: function for cyclic block coordinate descent over an active set, callable.
        active_set: indices of main effects to optimize over, a numpy int array.
        active_interaction_set: indices of interaction effects to optimize over, a numpy int array.
        B: B-spline transformed sparse matrices for main effects, list of sparse matrices of shapes [(N, Ki+1), ...].
        B_interaction: B-spline transformed sparse matrices for interaction effects, list of sparse matrices of shapes [(N, Kij+1), ...].
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
        
    Returns:
        optimal_solution_path: (beta_opt, delta_opt, zeta_opt, alpha_opt, tau_opt, J_opt, active_set_opt, active_interaction_set_opt, val_loss_opt).
        sparse_solution_path: (beta_sp, delta_sp, zeta_sp, alpha_sp, tau_sp, J_sp, active_set_sp, active_interaction_set_sp, val_loss_sp).
    """
    
    d = len(B)
    N = Y.shape[0]
    val_loss_opt = np.inf
    val_loss = np.inf*np.ones((taus.shape[0],),dtype=float)
    val_std_err = np.inf*np.ones((taus.shape[0],),dtype=float)
    sparsity = (active_set.shape[0]+active_interaction_set.shape[0])*np.ones((taus.shape[0],),dtype=float)
    J = np.zeros((taus.shape[0],),dtype=float)
    eps = 1e-8
    if eval_criteria == 'mse':
        evaluate = mean_squared_error
    elif eval_criteria == 'mae':
        evaluate = mean_absolute_error
    else:
        raise ValueError("Evaluation criteria {} is not supported".format(eval_criteria))
    
     
    # Generate b-splines for validation set for active set
    Bval = [None]*d
    for k in active_set:
        Bval[k] = sp.csr_matrix(dmatrix("bs(x, df={}, degree=3, include_intercept=False, lower_bound={}, upper_bound={})".format(K_main[k], Xmin[k], Xmax[k]), {"x": Xval[:,k]}),dtype=np.float64)
    Bval_interaction = [None]*len(interaction_terms)
    for k in active_interaction_set:
        f_i, f_j = interaction_terms[k]
        Bval_interaction[k] = sp.csr_matrix(dmatrix("te(bs(x1, df={}, degree=3, include_intercept=False, lower_bound={}, upper_bound={}), bs(x2, df={}, degree=3, include_intercept=False, lower_bound={}, upper_bound={}))".format(K_interaction[f_i], Xmin[f_i], Xmax[f_i], K_interaction[f_j], Xmin[f_j], Xmax[f_j]), {"x1": Xval[:,f_i], "x2": Xval[:,f_j]}),dtype=np.float64)

    # Tau path
    beta_HS = [deepcopy(beta)]*taus.shape[0]
    zeta_HS = [deepcopy(zeta)]*taus.shape[0]
    delta_HS = [deepcopy(delta)]*taus.shape[0]
    alpha_HS = [deepcopy(alpha)]*taus.shape[0]
        
    for i, tau in tqdm_notebook(enumerate(taus),desc='$\\tau$'):
        
        if i==0:
            beta_current = deepcopy(beta_HS[0])
            delta_current = deepcopy(delta_HS[0])
        else:
            beta_current = deepcopy(beta_HS[i-1])
            delta_current = deepcopy(delta_HS[i-1])
        
        if len(active_set)==0 and len(active_interaction_set)==0:
            Ypred = np.mean(Y)*np.ones(Y.shape,dtype=float)
        elif len(active_set)==0 and len(active_interaction_set)>0:
            Ypred = np.mean(Y)*np.ones(Y.shape,dtype=float)\
                    +np.array(sum([(B_interaction[k]).dot(delta_current[k]) for k in active_interaction_set])).reshape(Y.shape)
        elif len(active_set)>0 and len(active_interaction_set)==0:
            Ypred = np.mean(Y)*np.ones(Y.shape,dtype=float)\
                    +np.array(sum([(B[k]).dot(beta_current[k]) for k in active_set])).reshape(Y.shape)
        elif len(active_set)>0 and len(active_interaction_set)>0:
            Ypred = np.mean(Y)*np.ones(Y.shape,dtype=float)\
                    +np.array(sum([(B[k]).dot(beta_current[k]) for k in active_set])).reshape(Y.shape)\
                    +np.array(sum([(B_interaction[k]).dot(delta_current[k]) for k in active_interaction_set])).reshape(Y.shape)        
 
        z_max = np.max([np.max(zeta_HS[i]), np.max(alpha_HS[i])])
        zeta_HS[i] = np.where((zeta_HS[i]/z_max)>tau,
                              np.ones(zeta_HS[i].shape,dtype=float),
                              np.zeros(zeta_HS[i].shape,dtype=float))
        alpha_HS[i] = np.where((alpha_HS[i]/z_max)>tau,
                               np.ones(alpha_HS[i].shape,dtype=float),
                               np.zeros(alpha_HS[i].shape,dtype=float))
        Ypred, beta_HS[i], zeta_HS[i], delta_HS[i], alpha_HS[i] = CD_J_AS(Ypred = Ypred,
                                              beta = [deepcopy(beta_current), deepcopy(delta_current)],
                                              zeta = [zeta_HS[i], alpha_HS[i]],
                                              active_set = [np.where(zeta_HS[i][0,:]>tau)[0], np.where(alpha_HS[i][0,:]>tau)[0]],
                                              lam = [lam_1, 0.0],
                                              P = P, 
                                              P_interaction = P_interaction)
        train_loss = evaluate(y_scaler.inverse_transform(Y), y_scaler.inverse_transform(Ypred))


        if len(active_set)==0 and len(active_interaction_set)==0:
            Yvalpred = np.mean(Y)*np.ones(Yval.shape,dtype=float)
        elif len(active_set)==0 and len(active_interaction_set)>0:
            Yvalpred = np.mean(Y)*np.ones(Yval.shape,dtype=float)\
                       +np.array(sum([(Bval_interaction[k]).dot(delta_HS[i][k]) for k in active_interaction_set])).reshape(Yval.shape)
        elif len(active_set)>0 and len(active_interaction_set)==0:
            Yvalpred = np.mean(Y)*np.ones(Yval.shape,dtype=float)\
                       +np.array(sum([(Bval[k]).dot(beta_HS[i][k]) for k in active_set])).reshape(Yval.shape)
        elif len(active_set)>0 and len(active_interaction_set)>0:
            Yvalpred = np.mean(Y)*np.ones(Yval.shape,dtype=float)\
                       +np.array(sum([(Bval[k]).dot(beta_HS[i][k]) for k in active_set])).reshape(Yval.shape)\
                       +np.array(sum([(Bval_interaction[k]).dot(delta_HS[i][k]) for k in active_interaction_set])).reshape(Yval.shape)        
        val_loss[i] = evaluate(y_scaler.inverse_transform(Yval), y_scaler.inverse_transform(Yvalpred))
        val_std_err[i] = (mean_squared_error(y_scaler.inverse_transform(Yval), y_scaler.inverse_transform(Yvalpred))**0.5)/(Yval.shape[0]**0.5)
        sparsity[i] = np.count_nonzero(zeta_HS[i][0,:]) + np.count_nonzero(alpha_HS[i][0,:])
        J[i] = 0.5*mean_squared_error(Y, Ypred)+\
               lam_1*sum([(np.transpose(beta_HS[i][k])).dot(S[k].dot(beta_HS[i][k]))[0,0] for k in active_set])+\
               lam_1*sum([(np.transpose(delta_HS[i][k])).dot(S_interaction[k].dot(delta_HS[i][k]))[0,0] for k in active_interaction_set])+\
               eps*sum([np.dot(beta_HS[i][k][:,0],beta_HS[i][k][:,0]) for k in active_set])+\
               eps*sum([np.dot(delta_HS[i][k][:,0],delta_HS[i][k][:,0]) for k in active_interaction_set])+\
               lam_2*(np.sum(zeta_HS[i][0,:]))+\
               r*lam_2*(np.sum(alpha_HS[i][0,:]))    
        if logging ==True:
            with open(path+'/Training-HS.csv', "a") as f:
                f.write('{:.7f},{:.7f},{:.6f},{:.6f},{:.6f},{:.6f},{},{}\n'.format(lam_1,lam_2,tau,train_loss, val_loss[i], J[i],np.count_nonzero(zeta_HS[i][0,:]),np.count_nonzero(alpha_HS[i][0,:]))) 
        print('{:.7f},{:.7f},{:.6f},{:.6f},{:.6f},{:.6f},{},{}\n'.format(lam_1,lam_2,tau,train_loss, val_loss[i], J[i],np.count_nonzero(zeta_HS[i][0,:]),np.count_nonzero(alpha_HS[i][0,:])))
#             display(Math(r'\lambda_1: {:.6f}, \lambda_2: {:.6f}, Train-MAE: {:.6f}, Val-MAE: {:.6f}, Obj: {:.0f},'.format(lam_1,lam_2,train_loss, val_loss, J)+'\sum_{j \in S^c} z_j: '+'{},'.format(np.count_nonzero(zeta[j][0,:]))+'\sum_{ij \in S^c} z_{ij}: '+'{}.'.format(np.count_nonzero(alpha[j][0,:]))))
        df = pd.DataFrame(columns=[lam_1, lam_2, tau, *(zeta_HS[i][0,:])])
        with open(os.path.join(path, 'main_support_regularization_path.csv'), 'a') as f:
            df.to_csv(f, header=True, index=False)
        df = pd.DataFrame(columns=[lam_1, lam_2, tau, *(alpha_HS[i][0,:])])
        with open(os.path.join(path, 'interaction_support_regularization_path.csv'), 'a') as f:
            df.to_csv(f, header=True, index=False)
        if val_loss[i] <  val_loss_opt:
            val_loss_opt = deepcopy(val_loss[i])
            val_std_err_opt = deepcopy(val_std_err[i])
            beta_opt = deepcopy(beta_HS[i]) 
            zeta_opt = deepcopy(zeta_HS[i]) 
            delta_opt = deepcopy(delta_HS[i]) 
            alpha_opt = deepcopy(alpha_HS[i])
            active_set_opt = np.where(zeta_HS[i][0,:] == 1)[0] 
            active_interaction_set_opt = np.where(alpha_HS[i][0,:] == 1)[0]
            tau_opt = deepcopy(tau)        
            J_opt = deepcopy(J[i])
            
#     val_loss_percent = ((val_loss-val_loss_opt*np.ones((taus.shape[0],),dtype=float))/(val_loss_opt*np.ones((taus.shape[0],),dtype=float)))*100
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
    beta_sp = deepcopy(beta_HS[min_sparsity_min_val_index]) 
    zeta_sp = deepcopy(zeta_HS[min_sparsity_min_val_index]) 
    delta_sp = deepcopy(delta_HS[min_sparsity_min_val_index]) 
    alpha_sp = deepcopy(alpha_HS[min_sparsity_min_val_index])
    active_set_sp = np.where(zeta_HS[min_sparsity_min_val_index][0,:] == 1)[0]
    active_interaction_set_sp = np.where(alpha_HS[min_sparsity_min_val_index][0,:] == 1)[0] 
    tau_sp = deepcopy(taus[min_sparsity_min_val_index])
    J_sp = deepcopy(J[min_sparsity_min_val_index])
    
    return (beta_opt, delta_opt, zeta_opt, alpha_opt, tau_opt, J_opt, active_set_opt, active_interaction_set_opt, val_loss_opt), (beta_sp, delta_sp, zeta_sp, alpha_sp, tau_sp, J_sp, active_set_sp, active_interaction_set_sp, val_loss_sp)