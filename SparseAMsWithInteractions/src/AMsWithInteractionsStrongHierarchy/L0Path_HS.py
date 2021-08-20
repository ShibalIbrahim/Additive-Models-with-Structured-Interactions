from copy import deepcopy
from IPython.display import display
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

'''
    Sparse Additive Model fitting with cubic splines (bspline basis)
'''
def L0Path(MIP_HS = None,
           lam_1 = None,
           lams_2 = None,
           beta = None,
           zeta = None,
           delta = None,
           alpha = None,
           start = None,
           Taupath = None,
           active_set = None,
           active_interaction_set = None,
           B = None,
           B_interaction = None,
           BT_B = None,
           B_interactionT_B_interaction = None,
           Y = None,
           S = None,
           S_interaction = None,
           interaction_terms = None,
           path = None,
           r = None,
           logging = False):
    """Hyperparameter grid search for L0 penalty for nonparametric additive models with interactions under strong hierarchy
    
    Args:
        lam_1: smoothness penalty for b-splines, float scaler.
        lams_2: L0 penalty for b-splines, array of float scalers.
        MIP_HS: function that solves convex relaxtion of the MIP under Strong Hierarchy, callable.
        beta: coefficients for main effects, list of arrays of shapes [(Ki+1, 1), ...].
        zeta: binary vector to track which main effects are in the active set, a bool array of shape (1, d)
            corresponds to z_i's in the paper.
        delta: coefficients for interaction effects, list of arrays of shapes [(Kij+1, 1), ...].
            corresponds to theta in the paper.
        alpha: binary vector to track which interactions effects are in the active interaction set, a bool array of shape (1, Imax)
            corresponds to z_ij's in the paper.
        start: used for warm-starting, int scaler.
        Taupath: a function that thresholds to generate subset of main/interaction feasible subsets that maintain
            strong hierarchy and solves on those sets, callable.
        active_set: indices of main effects to optimize over, a numpy int array.
        active_interaction_set: indices of interaction effects to optimize over, a numpy int array.
        B: B-spline transformed sparse matrices for main effects, list of sparse matrices of shapes [(N, Ki+1), ...].
        B_interaction: B-spline transformed sparse matrices for interaction effects, list of sparse matrices of shapes [(N, Kij+1), ...].
        BT_B: B^T*B matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
        B_interactionT_B_interaction: B^T*B matrices for interaction effects, list of sparse matrices of shapes [(Kij+1, Kij+1), ...].
        Y: training target responses, a float numpy array of shape (N,).
        S: Smoothness matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
        S_interaction: Smoothness matrices for interaction effects, list of sparse matrices of shapes [(Kij+1, Kij+1), ...].
        interaction_terms: list of interaction effects to consider if only a subset need to be considered, 
            a 2D numpy array of of shape (Imax, 2).
        path: folder path to log results to, str.
        r: relative scaling factor for L0 penalty between main and interaction effects.
            We consider r=1.0 (corresponds to alpha symbol in the paper), float scaler. 
        logging: whether to log results to a file, bool scaler.
        
    Returns:
        parameters_path: (beta, delta, zeta, alpha).
        optimal_solution_path: (beta_opt, delta_opt, zeta_opt, alpha_opt, lam_1_opt, lam_2_opt, tau_opt, J_opt, active_set_opt, active_interaction_set_opt, val_loss_opt).
        sparse_solution_path: (beta_sp, delta_sp, zeta_sp, alpha_sp, lam_1_sp, lam_2_sp, tau_sp, J_sp, active_set_sp, active_interaction_set_sp, val_loss_sp).
    """
    
    d = len(B)
    N = Y.shape[0]
    val_loss_opt = np.inf
    val_loss = np.inf*np.ones((lams_2.shape[0],),dtype=float)
    sparsity = (d+np.floor(comb(d, 2, exact=False)))*np.ones((lams_2.shape[0],),dtype=float)
    J = np.zeros((lams_2.shape[0],),dtype=float)
    eps = 1e-8
    
    P = [None]*d
    for k in active_set:
        P[k] = sp.linalg.splu(BT_B[k]+2*N*(lam_1*S[k]+eps*sp.csr_matrix(np.identity(B[k].shape[1]))))
    P_interaction = [None]*len(interaction_terms)
    for k in active_interaction_set:
        P_interaction[k] = sp.linalg.splu(B_interactionT_B_interaction[k]+2*N*(lam_1*S_interaction[k]+eps*sp.csr_matrix(np.identity(B_interaction[k].shape[1]))))
    
    # L0 path
    for j, lam_2 in tqdm_notebook(enumerate(lams_2), desc='$\lambda_2$'):
        
        if start==0:            
            if j==0:
                beta_current = deepcopy(beta[0])
                zeta_current = deepcopy(zeta[0])
                delta_current = deepcopy(delta[0])
                alpha_current = deepcopy(alpha[0])
            else:
                beta_current = deepcopy(beta[j-1])
                zeta_current = deepcopy(zeta[j-1])
                alpha_current = deepcopy(alpha[j-1])
                delta_current = deepcopy(delta[j-1])
        else:
            beta_current = deepcopy(beta[j])
            zeta_current = deepcopy(zeta[j])
            alpha_current = deepcopy(alpha[j])
            delta_current = deepcopy(delta[j])

        Ypred = np.mean(Y)*np.ones(Y.shape,dtype=float)
            
        '''
            Hierarchical Sparsity
        '''
        Ypred, beta[j], zeta[j], delta[j], alpha[j] = MIP_HS(Ypred = deepcopy(Ypred), beta = [deepcopy(beta_current), deepcopy(delta_current)], zeta = [deepcopy(zeta_current), deepcopy(alpha_current)], lam = [lam_1, lam_2])
        
        optimal_solution_path, sparse_solution_path = Taupath(
            lam_1 = lam_1,
            lam_2 = lam_2,
            beta = deepcopy([np.zeros(bb.shape,dtype=float) for bb in beta[j]]),
            zeta = deepcopy(zeta[j]),
            delta = deepcopy([np.zeros(bb.shape,dtype=float) for bb in delta[j]]),
            alpha = deepcopy(alpha[j]),
            P = P, 
            P_interaction = P_interaction)
        
        beta_opt_path, delta_opt_path, zeta_opt_path, alpha_opt_path, tau_opt_path, J_opt_path, active_set_opt_path, active_interaction_set_opt_path, val_loss_opt_path = optimal_solution_path
        beta_sp_path, delta_sp_path, zeta_sp_path, alpha_sp_path, tau_sp_path, J_sp_path, active_set_sp_path, active_interaction_set_sp_path, val_loss_sp_path = sparse_solution_path
        if val_loss_opt_path <  val_loss_opt:
            val_loss_opt = deepcopy(val_loss_opt_path)
            beta_opt = deepcopy(beta_opt_path) 
            zeta_opt = deepcopy(zeta_opt_path) 
            delta_opt = deepcopy(delta_opt_path) 
            alpha_opt = deepcopy(alpha_opt_path)
            active_set_opt = deepcopy(active_set_opt_path)
            active_interaction_set_opt = deepcopy(active_interaction_set_opt_path)
            tau_opt = deepcopy(tau_opt_path)
            lam_1_opt = deepcopy(lam_1)
            lam_2_opt = deepcopy(lam_2)
            J_opt = deepcopy(J_opt_path)
            val_loss_sp = deepcopy(val_loss_sp_path)
            beta_sp = deepcopy(beta_sp_path) 
            zeta_sp = deepcopy(zeta_sp_path) 
            delta_sp = deepcopy(delta_sp_path) 
            alpha_sp = deepcopy(alpha_sp_path)
            active_set_sp = deepcopy(active_set_sp_path)
            active_interaction_set_sp = deepcopy(active_interaction_set_sp_path)
            tau_sp = deepcopy(tau_sp_path)
            lam_1_sp = deepcopy(lam_1)
            lam_2_sp = deepcopy(lam_2)
            J_sp = deepcopy(J_sp_path)

    
    return (beta, delta, zeta, alpha), (beta_opt, delta_opt, zeta_opt, alpha_opt, lam_1_opt, lam_2_opt, tau_opt, J_opt, active_set_opt, active_interaction_set_opt, val_loss_opt), (beta_sp, delta_sp, zeta_sp, alpha_sp, lam_1_sp, lam_2_sp, tau_sp, J_sp, active_set_sp, active_interaction_set_sp, val_loss_sp)