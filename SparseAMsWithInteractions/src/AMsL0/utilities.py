"""Utilities for generating b-splines and quadratic penalties"""
from copy import deepcopy
from functools import partial
from IPython.display import Math
from ipywidgets import *
import multiprocessing as mp
import numpy as np
import os
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error
from patsy import dmatrix
from tqdm import notebook
import warnings

def bspline_batch(interaction_terms_batch = None, X=None, Xmin=None, Xmax=None, K_interaction = None, eps = None):
    B = [sp.csr_matrix(dmatrix("te(bs(x1, df={}, degree=3, include_intercept=False, lower_bound={}, upper_bound={}), bs(x2, df={}, degree=3, include_intercept=False, lower_bound={}, upper_bound={}))".format(K_interaction[f_i], Xmin[f_i]-eps, Xmax[f_i]+eps, K_interaction[f_j], Xmin[f_j]-eps, Xmax[f_j]+eps), {"x1": X[:,f_i], "x2": X[:,f_j]})) for f_i, f_j in interaction_terms_batch]
    return B

def screening(main_terms_batch=None, X=None, Y=None, Xmin=None, Xmax=None, Ki=None, degree=3, eps=1e-6):
    """Screens main terms according to their marginal residuals
    """
    res = []
    n = Y.shape[0]
    for j in main_terms_batch:
        B = sp.csr_matrix(dmatrix("bs(x, df={}, degree={}, include_intercept=False, lower_bound={}, upper_bound={})".format(
                Ki, degree, Xmin[j], Xmax[j]), {"x": X[:,j]}))
        b = sp.linalg.spsolve((B.transpose()).dot(B) + n * eps * sp.csr_matrix(np.identity(B.shape[1])),(B.transpose()).dot(Y)).reshape(-1,1)
        res.append(mean_squared_error(Y, B.dot(b)))
        del B
    return res        

'''
    Cubic splines (bspline basis)
'''
def generate_bspline_transformed_X(X,
                                   Xmin,
                                   Xmax,
                                   Ki,
                                   main_terms,
                                   degree=3):
    """Generates B-spline transformations for main effects.

    Args:
        X: data matrix, float numpy array of shape (N, p).
        Xmin: Minimum value per covariate, float numpy array of shape (p, ).
        Xmax: Maximum value per covariate, float numpy array of shape (p, ).
        Ki: Degrees of freedom for b-spline basis, int scalar.
        degree: degree of b-spline basis, int scalar.

    Returns:
        Btrain: B-spline transformed sparse matrices for main effects, list of sparse matrices of shapes [(N, Ki+1), ...].
        K_main: Degrees of freedom for each main effect after accounting for minimum ranks of b-spline transformed matrices,
            a numpy array of shape (p, ).
    """
    N, p = X.shape
    
    K_main = []
    for j in range(p):
        num_unique_covariate = np.unique(X[:, j], axis=0).shape[0]
        if num_unique_covariate>Ki:
            K_main.append(Ki)
        else:
            K_main.append(num_unique_covariate)
    
    K_main = np.clip(K_main, a_min=degree, a_max=None) 
    print('Generating bspline basis for main effects...')
    Btrain = [sp.csr_matrix(
        np.array(
            dmatrix("bs(x, df={}, degree={}, include_intercept=False, lower_bound={}, upper_bound={})".format(
                K_main[j], degree, Xmin[j], Xmax[j]), {"x": X[:,j]}
            )
        ), dtype=np.float64) for j in range(p)] 
    K_main = [np.linalg.matrix_rank(B.toarray())-1 for B in Btrain]
    K_main = np.clip(K_main, a_min=degree, a_max=None) 
    Btrain = [sp.csr_matrix(
        np.array(
            dmatrix("bs(x, df={}, degree={}, include_intercept=False, lower_bound={}, upper_bound={})".format(
                K_main[j], degree, Xmin[j], Xmax[j]), {"x": X[:,j]}
            )
        ), dtype=np.float64) for j in notebook.trange(p, desc='$B_{train}$')] 
    print('Bspline basis for main effects generated.')
            
    return Btrain, K_main


def generate_bspline_quadratic_penalties(K_main):
    """Generate Quadratic penalties for main and interaction effects.
    
    Args:
        K_main: Degrees of freedom for each main effect after accounting for minimum ranks of b-spline transformed matrices,
            a numpy array of shape (p, ).
    
    Returns:
        S_main: Quadratic penalty for main effects,
            list of sparse matrices of shapes [(Ki-2, Ki), ... ].         
    """
    # Main Effects
    S_main = []
    for j in range(len(K_main)):
        D = sp.csc_matrix(sp.diags([-1, 2, -1], [0, 1, 2], shape=(K_main[j] - 2, K_main[j])))
        S = (D.transpose()).dot(D)
        S = sp.csc_matrix(
            np.append(np.zeros((S.shape[0]+1, 1), dtype=float),\
            np.append(np.zeros((1, S.shape[1]), dtype=float), S.toarray(), axis=0), axis=1)
        )
        S_main.append(S)
            
    return S_main 


# Inputs: 1) Bspline transformed X matrix: B
#         2) Output y, 
#         3) Difference penalty matrix: S
#         4) Hyperparameter: lam
# Outputs: 1) beta, zeta

def solve(B = None,
          BT_B = None,
          P = None,
          y = None,
          beta = None,
          S = None,
          lam = [1, 1],
          feature = None,
          eps = 1e-8):
    """
    Args: 
        B: Bspline transformed X matrix, float sparse matrix of shape (N, K).
        BT_B: B^T*B, float sparse matrix of shape (K, K).
        P: LU factors of B^T*B + lam_sm*S, superlu object.
        y: residual, float numpy array of shape (N, 1).
        beta: parameters, float numpy array of shape (K, ).
        S: Quadratic smoothness penalty fos shape (K-2, K).
        lam: smoothness penalty, float scaler.
        feature: covariate index, int scaler.
        eps: numerical stability (account for in the objective), float scaler.
        
    Returns:
        beta: parameters, float numpy array of shape (K, ).
        zeta: whether effect is zero or nonzero, boolean numpy array of shape (1, ).    
    """
        
    N = y.shape[0]
    K = B.shape[1]
    J_0 = 0.5*mean_squared_error(y, np.zeros((N,1),dtype=float))

#    b, _ = sp.linalg.cg(A = BT_B+2*N*lam[0]*S, b = (B.transpose()).dot(y), x0 = deepcopy(beta), M = P, tol = cg_tol)
    b = P.solve((B.transpose()).dot(y))
    b = b.reshape(-1,1)
    if np.sum(np.isnan(b))>0:
        b = np.zeros((K,1), dtype=float)
        J_1 = np.inf
    else:
        J_1 = 0.5 * mean_squared_error(y, (B).dot(b))+\
              lam[0] * (np.transpose(b).dot(S.dot(b)))[0,0]+\
              eps * np.dot(b[:, 0], b[:, 0])+\
              lam[1]

    zeta = np.zeros((1, ), dtype=bool)
    if J_0<=J_1:
        beta = np.zeros((K, 1), dtype=float)
        case = 0
    elif J_1<J_0:
        beta = b.copy()
        zeta[0] = True
        case = 1
    return beta, zeta