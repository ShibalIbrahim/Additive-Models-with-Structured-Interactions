from copy import deepcopy
from IPython.display import Math
from IPython import display
from ipywidgets import *
import numpy as np
import scipy.sparse as sp
from scipy.special import comb
from sklearn.metrics import mean_squared_error
from tqdm import tnrange, tqdm_notebook

from SparseAMsWithInteractions.src.AMsL0.utilities import solve

def CD_Separate_ActiveSet(Ypred=None,
                          B=None,
                          BT_B=None,
                          P=None,
                          S=None,
                          I=None,
                          beta=None,
                          zeta=None,
                          lam=None,
                          active_set=None,
                          Y=None,
                          main_terms=None,
                          max_iter=None,
                          tol=1e-4,
                          verbose=False,
                          path=None):

    """Cyclic Block Coordinate Descent over active set.
        
    Args:
        Ypred: current prediction, numpy array of shape (N, ).
        B: B-spline transformed sparse matrices for main effects, list of sparse matrices of shapes [(N, Ki+1), ...].
        BT_B: B^T*B matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
        P: B^T*B + 2*N*(lam_1*S + eps*I) matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
            eps is a small epsilon for numerical stability.
        S: Smoothness matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
        I: number of possible main effects equal to d, int scaler.
        beta: coefficients for main effects, list of arrays of shapes [(Ki+1, 1), ...].
        zeta: binary vector to track which main effects are in the active set, a bool array of shape (1, d)
        lam: regularization parameters [lam_1, lam_2], list of floats.
        active_set: indices of main effects to optimize over, a numpy int array.
        Y: training target responses, a float numpy array of shape (N,).
        main_terms: list of main effects to consider if only a subset need to be considered, 
            not supported yet.
        max_iter: maximum number of Cyclic BCD on the active set, int scaler.
        tol: relative loss termination criteria for stopping, a float scalar.
        verbose: for printing optimization steps, bool scaler.
        path: for logging, str.
        
    Returns:
        Ypred: Updated prediction, numpy array of shape (N, ).
        beta: Updated coefficients for main effects, list of arrays of shapes [(Ki+1, 1), ...].
        zeta: Updated binary vector to track which main effects are in the active set, a bool array of shape (1, d)
        
    """
    N = Y.shape[0]
    J_prev = 0.0
    J = 0.5*mean_squared_error(Y, Ypred)+\
        lam[0]*sum([(np.transpose(beta[k])).dot(S[k].dot(beta[k]))[0,0] for k in active_set])+\
        lam[1]*(np.count_nonzero(zeta[0,:]))
    if verbose == True:
        display(Math(r'Input~Obj: {:.0f}, '.format(J)+'\sum_{j \in S^c} z_j = '+'{} \leq {}.'.format(np.count_nonzero(zeta[0,:]), len(active_set)))) 

    for it in range(max_iter):
        for j in active_set:
            
            ## Residual updates
            Ypred -= B[j].dot(beta[j])
            res = Y-Ypred
            beta[j], zeta[:,j] = solve(B=B[j], BT_B=BT_B[j], P=P[j], y=res, S=S[j], lam=lam)
            Ypred += B[j].dot(beta[j])

        # Check convergence
        J_prev = J
        J = 0.5*mean_squared_error(Y, Ypred)+\
            lam[0]*sum([(np.transpose(beta[k])).dot(S[k].dot(beta[k]))[0,0] for k in active_set])+\
            lam[1]*(np.count_nonzero(zeta[0,:]))       
        J_del = J-J_prev
        if np.absolute(J_del/J)<tol:
            break
            
    if verbose == True:
        display(Math(r'Output~Obj: {:.0f}, |\Delta J/J|: {:.4f}, '.format(J, np.absolute(J_del/J))+'\sum_{j \in S^c} z_j: '+'{} \leq {}.'.format(np.count_nonzero(zeta[0,:]), len(active_set)))) 
    if(it == max_iter-1):
        print('Warning: CD over active set did not converge within the chosen max_iter!')
        with open(path+'/Warning.txt', "a") as f:
            f.write('Warning: CD over active set did not converge within the chosen max_iter!')
    return Ypred, beta, zeta

def CD_Separate(CD_S_AS=None,
                Ypred=None,
                B=None,
                BT_B=None,
                P=None,
                S=None,
                I=None,
                beta=None,
                zeta=None,
                lam=None,
                active_set=None,
                full_set=None,
                Y=None,
                main_terms=None,
                max_iter=None,
                tol=1e-4,
                active_set_update=True,
                MaxSuppSize=None,
                verbose=False,
                path=None):
    """Cyclic Block Coordinate Descent over the complete set of main effects.
    
    Args:
        CD_S_AS: a callable function that optimizes over a reduced set of main effects, callable.
        Ypred: numpy array of shape (N, ).
        B: B-spline transformed sparse matrices for main effects, list of sparse matrices of shapes [(N, Ki+1), ...].
        BT_B: B^T*B matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
        P: B^T*B + 2*N*(lam_1*S + eps*I) matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
            eps is a small epsilon for numerical stability.
        S: Smoothness matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
        I: number of possible main effects equal to d, int scaler.
        beta: coefficients for main effects, list of arrays of shapes [(Ki+1, 1), ...].
        zeta: binary vector to track which main effects are in the active set, a bool array of shape (1, d)
        lam: regularization parameters [lam_1, lam_2], list of floats.
        active_set: indices of main effects to optimize over, a numpy int array.
        full_set: indices of all main effects, a numpy int array.
        Y: training target responses, a float numpy array of shape (N,).
        main_terms: list of main effects to consider if only a subset need to be considered, 
            not supported yet.
        max_iter: maximum number of Cyclic BCD on the active set, int scaler.
        tol: relative loss termination criteria for stopping, a float scalar.
        active_set_update: update active set for next setting of lambda in regularization path, bool scaler.
        MaxSuppSize: Stop L0 regularization if the active set is larger than the MaxSuppSize
            and move to next smoothing lambda setting and start L0 regularization, int scaler.
        verbose: for printing optimization steps, bool scaler.
        path: for logging, str.
    
    Returns:
        Ypred: Updated prediction, numpy array of shape (N, ).
        beta: Updated coefficients for main effects, list of arrays of shapes [(Ki+1, 1), ...].
        zeta: Updated binary vector to track which main effects are in the active set, a bool array of shape (1, d).
        active_set: Updated indices of nonzero main effects, a numpy int array.
        MaxSuppSize_flag: indicates Maximum Support size is reached, bool scaler.
        
    """
    N = Y.shape[0]    
    MaxSuppSize_flag = 0 
    res = Y-Ypred 
    beta_p = []
    for b, bT_b, s in zip(B, BT_B, S):
        try:
            beta_p.append(sp.linalg.spsolve(bT_b+2*N*lam[0]*s,(b.transpose()).dot(res)).reshape(-1,1))
        except:
            beta_p.append(np.zeros((b.shape[1],1),dtype=float))
    res_p = np.array([np.linalg.norm(res-b.dot(bp)) for b, bp in zip(B, beta_p)])
    
    ## Active set updated after each hyperparameter setting on the L0 path.
    if active_set_update:
        if active_set is None: 
            A = int(np.ceil(0.01*I))
            active_set = res_p.argsort()[:A]        
        else:
            A = np.minimum(int(np.ceil(0.2*len(active_set))), int(np.ceil(0.01*I)))
            active_set = np.union1d(active_set, res_p.argsort()[:A]) 
    else:
        active_set = res_p.argsort()
    
    '''
        Coordinate Descent over full set
    '''
    for it in range(max_iter):
        
        # Optimize over active set
        Ypred, beta, zeta = CD_S_AS(Ypred=Ypred,
                                    B=B,
                                    BT_B=BT_B,
                                    P=P,
                                    S=S,
                                    I=I,
                                    beta=beta,
                                    zeta=zeta,
                                    lam=lam,
                                    active_set=active_set)
        if len(np.where(zeta[0,:] == 1)[0]) > MaxSuppSize:
            MaxSuppSize_flag = 1
            break
        J = 0.5*mean_squared_error(Y, Ypred)+\
            lam[0]*sum([(np.transpose(beta[k])).dot(S[k].dot(beta[k]))[0,0] for k in active_set])+\
            lam[1]*(np.count_nonzero(zeta[0,:]))

        if verbose == True:
            display(Math(r'Iteration: {}, Obj: {:.0f}, '.format(it, J)+', \sum_{j \in S^c} z_j: '+'{} \leq {}.'.format(np.count_nonzero(zeta[0,:]), len(active_set)))) 
        
        # Checking for violations outside active set
        for j in [x for x in full_set if x not in active_set]:
            Ypred -= B[j].dot(beta[j])
            res = Y - Ypred
            beta[j], zeta[:,j] = solve(B=B[j], BT_B=BT_B[j], P=P[j], y=res, S=S[j], lam=lam)
            Ypred += B[j].dot(beta[j])
            
        # Stop if active set doesn't change.
        if np.count_nonzero(zeta[0,:])==active_set.shape[0]:
            if np.sum(sorted(active_set) == np.where(zeta[0,:] == 1)[0])==active_set.shape[0]:
                #print('Active set converged')
                break
        active_set = np.where(zeta[0,:] == 1)[0]
    if(it == max_iter-1):
        print('Warning: CD over full set did not converge within the chosen max_iter!')
        with open(path+'/Warning.txt', "a") as f:
            f.write('Warning: CD over full set did not converge within the chosen max_iter!')

    return Ypred, beta, zeta, active_set, MaxSuppSize_flag