from copy import deepcopy
from functools import partial
from IPython.display import Math
from ipywidgets import *
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import multiprocessing as mp
import numpy as np
import os
from patsy import dmatrix
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error
from tqdm import notebook
import warnings

os.environ['QT_QPA_PLATFORM']='offscreen'
font = {'weight' : 'bold',
        'size'   : 14}

def bspline_batch(interaction_terms_batch = None, X=None, Xmin=None, Xmax=None, K_interaction = None, eps = None):
    B = [sp.csr_matrix(dmatrix("te(bs(x1, df={}, degree=3, include_intercept=False, lower_bound={}, upper_bound={}), bs(x2, df={}, degree=3, include_intercept=False, lower_bound={}, upper_bound={}))".format(K_interaction[f_i], Xmin[f_i]-eps, Xmax[f_i]+eps, K_interaction[f_j], Xmin[f_j]-eps, Xmax[f_j]+eps), {"x1": X[:,f_i], "x2": X[:,f_j]})) for f_i, f_j in interaction_terms_batch]
    return B



def screening(interaction_terms_batch=None, X=None, Y=None, Xmin=None, Xmax=None, Kij=None, degree=3, eps=1e-6):
    """Screens interaction terms according to their marginal residuals
    """
    res = []
    n = Y.shape[0]
    for f_i, f_j in interaction_terms_batch:
        B = sp.csr_matrix(dmatrix("te(bs(x1, df={}, degree={}, include_intercept=False, lower_bound={}, upper_bound={}), bs(x2, df={}, degree={}, include_intercept=False, lower_bound={}, upper_bound={}))".format(Kij, degree, Xmin[f_i], Xmax[f_i], Kij, degree, Xmin[f_j], Xmax[f_j]), {"x1": X[:,f_i], "x2": X[:,f_j]}))
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
                                   Kij,
                                   interaction_terms,
                                   degree=3):
    """Generates B-spline transformations for main and interaction effects.

    Args:
        X: data matrix, float numpy array of shape (N, p).
        Xmin: Minimum value per covariate, float numpy array of shape (p, ).
        Xmax: Maximum value per covariate, float numpy array of shape (p, ).
        Ki: Degrees of freedom for b-spline basis, int scalar.
        Kij: Degrees of freedom for b-spline basis in each covariate direction, int scalar.
        interaction_terms: interaction effects, int numpy 2D array containing pairwise indices of
            interactions to be considered.
        degree: degree of b-spline basis, int scalar.

    Returns:
        Btrain: B-spline transformed sparse matrices for main effects, list of sparse matrices of shapes [(N, Ki+1), ...].
        Btrain_interaction: B-spline transformed sparse matrices of shape list of sparse matrices of shapes [(N, Kij*Kij+1), ...].
        K_main: Degrees of freedom for each main effect after accounting for minimum ranks of b-spline transformed matrices,
            a numpy array of shape (p, ).
        K_interaction: Degrees of freedom for each interaction effect after accounting for minimum ranks of b-spline 
            transformed matrices of main effects, a numpy array of shape (len(interaction_terms), )
    """
    N, p = X.shape
    
    K_main = []
    for j in range(p):
        num_unique_covariate = np.unique(X[:, j], axis=0).shape[0]
        if num_unique_covariate>Ki:
            K_main.append(Ki)
        else:
            K_main.append(num_unique_covariate)
    
    
    print('Generating bspline basis for main effects...')
    Btrain = [sp.csr_matrix(
        np.array(
            dmatrix("bs(x, df={}, degree={}, include_intercept=False, lower_bound={}, upper_bound={})".format(
                K_main[j], degree, Xmin[j], Xmax[j]), {"x": X[:,j]}
            )
        ), dtype=np.float64) for j in range(p)] 
    K_main = [np.linalg.matrix_rank(B.toarray())-1 for B in Btrain]
    K_main = np.clip(K_main, a_min=degree, a_max=None) 
    K_interaction = np.clip(K_main, a_min=degree, a_max=Kij) 
    Btrain = [sp.csr_matrix(
        np.array(
            dmatrix("bs(x, df={}, degree={}, include_intercept=False, lower_bound={}, upper_bound={})".format(
                K_main[j], degree, Xmin[j], Xmax[j]), {"x": X[:,j]}
            )
        ), dtype=np.float64) for j in notebook.trange(p, desc='$B_{train}$')] 
    print('Bspline basis for main effects generated.')
        
    print('Generating bspline basis for interaction effects...')    
    Btrain_interaction = [sp.csr_matrix(
        np.array(
            dmatrix("te(bs(x1, df={}, degree={}, include_intercept=False, lower_bound={}, upper_bound={}), bs(x2, df={}, degree={}, include_intercept=False, lower_bound={}, upper_bound={}))".format(
                K_interaction[f_i], degree, Xmin[f_i], Xmax[f_i], K_interaction[f_j], degree, Xmin[f_j], Xmax[f_j]), {"x1": X[:, f_i], "x2": X[:, f_j]}
            )
        ), dtype=np.float64) for f_i, f_j in notebook.tqdm(interaction_terms, desc='$B_{train-interaction}$')]
    print('Bspline basis for interaction effects generated')
    
    return Btrain, Btrain_interaction, K_main, K_interaction


def generate_bspline_quadratic_penalties(K_main,
                                         K_interaction,
                                         interaction_terms):
    """Generate Quadratic penalties for main and interaction effects.
    
    Args:
        K_main: Degrees of freedom for each main effect after accounting for minimum ranks of b-spline transformed matrices,
            a numpy array of shape (p, ).
        K_interaction: Degrees of freedom for each interaction effect after accounting for minimum ranks of b-spline 
            transformed matrices of main effects, a numpy array of shape (len(interaction_terms), )
        interaction_terms: interaction effects, int numpy 2D array containing pairwise indices of
            interactions to be considered.
    
    Returns:
        S_main: Quadratic penalty for main effects,
            list of sparse matrices of shapes [(Ki-2, Ki), ... ]. 
        S_interaction: Quadratic penalty for main interaction effects,
            list of sparse matrices of shapes [(Kij*Kij-2, Kij*Kij),  ...].
        
    """
    # Main Effects
    S_main = []
    for j in range(len(K_main)):
        D = sp.csc_matrix(sp.diags([-1, 2, -1], [0, 1, 2], shape=(K_main[j] - 2, K_main[j])))
        S = (D.transpose()).dot(D)
        S = sp.csc_matrix(np.append(np.zeros((S.shape[0]+1,1),dtype=float),\
                      np.append(np.zeros((1,S.shape[1]),dtype=float),S.toarray(),axis=0),\
                      axis=1))
        S_main.append(S)
    
    # Interaction Effects
    S_interaction = []
    for (i,j) in interaction_terms:
        Di = sp.csc_matrix(sp.diags([-1, 2, -1], [0, 1, 2], shape=(K_interaction[i] - 2, K_interaction[i])))
        S_i = (Di.transpose()).dot(Di)        
        Dj = sp.csc_matrix(sp.diags([-1, 2, -1], [0, 1, 2], shape=(K_interaction[j] - 2, K_interaction[j])))
        S_j = (Dj.transpose()).dot(Dj)
        
        S_interaction_i = sp.csc_matrix(
            np.append(
                np.zeros((int(S_i.shape[0] * S_j.shape[1]) + 1, 1), dtype=float),
                np.append(
                    np.zeros((1, int(S_i.shape[0] * S_j.shape[1])), dtype=float),
                    np.kron(S_i.toarray(), np.identity(K_interaction[j])),
                    axis=0
                ),
                axis=1
            )
        )
        S_interaction_j = sp.csc_matrix(
            np.append(
                np.zeros((int(S_i.shape[0] * S_j.shape[1]) + 1,1), dtype=float),
                np.append(
                    np.zeros((1, int(S_i.shape[0] * S_j.shape[1])), dtype=float), 
                    np.kron(np.identity(K_interaction[i]),S_j.toarray()), axis=0),
                    axis=1
                )
            )
        S_interaction.append(S_interaction_i + S_interaction_j)
        
    return S_main, S_interaction 


def solve(B=None,
          P=None,
          y=None,
          beta=None,
          S=None,
          lam=None,
          eps=1e-8):
    """
    Args: 
        B: Bspline transformed X matrix, float sparse matrix of shape (N, K).
        P: LU factors of B^T*B + lam_sm*S, superlu object.
        y: residual, float numpy array of shape (N, 1).
        S: Quadratic smoothness penalty fos shape (K-2, K).
        lam: smoothness penalty, float scaler.
        eps: numerical stability (account for in the objective), float scaler.
        
    Returns:
        beta: parameters, float numpy array of shape (K, ).
        zeta: whether effect is zero or nonzero, boolean numpy array of shape (1, ).    
    """
    
    N = y.shape[0]
    K = B.shape[1]
    J_0 = 0.5*mean_squared_error(y, np.zeros_like(y))

    b = P.solve((B.transpose()).dot(y))
    b = b.reshape(-1,1)
#     if np.sum(np.isnan(b))>0:
#         b = np.zeros((K,1),dtype=float)
#         J_1 = np.inf
#     else:
    J_1 = 0.5*mean_squared_error(y, B.dot(b))+\
          lam[0]*((np.transpose(b)).dot(S.dot(b)))[0,0]+\
          eps*np.dot(b[:, 0], b[:, 0])+\
          lam[1]

    zeta = np.zeros((1, ), dtype=bool)
    if J_0<=J_1:
        beta = np.zeros((K,1),dtype=float)
        case = 0
    elif J_1<J_0:
        beta = deepcopy(b)
        zeta[0] = True
        case = 1
    return beta, zeta
