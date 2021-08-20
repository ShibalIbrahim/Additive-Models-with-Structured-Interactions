from copy import deepcopy
from IPython.display import Math
from ipywidgets import *
import numpy as np
import scipy.sparse as sp
from scipy.special import comb
from sklearn.metrics import mean_squared_error
from tqdm import notebook
import warnings

from SparseGAMsWithInteractions.src.GAMsWithInteractionsL0 import utilities

def CD_Joint_ActiveSet(Ypred = None,
                       beta = None,
                       zeta = None,
                       active_set = None,
                       lam = None,
                       P = None,
                       P_interaction = None,
                       Y = None,
                       B = None,
                       B_interaction = None,
                       S = None,
                       S_interaction = None,
                       I = None,
                       interaction_terms = None,
                       r = None,
                       max_iter = None,
                       tol = 1e-4,
                       verbose=False,
                       path = None):
    """Cyclic Block Coordinate Descent over active set.
        
    Args:
        Ypred: current prediction, numpy array of shape (N, ).
        beta: coefficients for main/interaction effects, 2 lists of arrays of shapes [ [(Ki+1, 1), ...], [(Kij+1, 1), ...]]
        zeta: binary vector to track which main effects are in the active set, 2 bool arrays of shape [(1, d), (1, Imax)]
        active_set: indices of main effects to optimize over, a numpy int array.
        lam: regularization parameters [lam_1, lam_2], list of floats.
        P: B^T*B + 2*N*(lam_1*S_i + eps*I) matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
            eps is a small epsilon for numerical stability.
        P_interaction: B^T*B + 2*N*(lam_1*S_ij + eps*I) matrices for main effects, list of sparse matrices of shapes [(Kij+1, Kij+1), ...].
            eps is a small epsilon for numerical stability.
        B: B-spline transformed sparse matrices for main effects, list of sparse matrices of shapes [(N, Ki+1), ...].
        B_interaction: B-spline transformed sparse matrices for interaction effects, list of sparse matrices of shapes [(N, Kij+1), ...].
        S: Smoothness matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
        S_interaction: Smoothness matrices for interaction effects, list of sparse matrices of shapes [(Kij+1, Kij+1), ...].
        I: number of possible main and interaction effects, int scalers.
        Y: training target responses, a float numpy array of shape (N,).
        interaction_terms: list of interaction effects to consider if only a subset need to be considered, 
            a 2D numpy array of of shape (Imax, 2).
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
    delta = beta[1]
    beta = beta[0]
    alpha = zeta[1]
    zeta = zeta[0]
    active_interaction_set = active_set[1]
    active_set = active_set[0]
    Bspam = B
    Bspam_interaction = B_interaction
    Pspam = P
    Pspam_interaction = P_interaction
    d = I[0]
    dinteraction = I[1]
    debugging_mode = False
    eps = 1e-8
    J = 0.5*mean_squared_error(Y, Ypred)+\
        lam[0]*sum([(np.transpose(beta[k])).dot(S[k].dot(beta[k]))[0,0] for k in active_set])+\
        lam[0]*sum([(np.transpose(delta[k])).dot(S_interaction[k].dot(delta[k]))[0,0] for k in active_interaction_set])+\
        eps*sum([np.dot(beta[k][:,0],beta[k][:,0]) for k in active_set])+\
        eps*sum([np.dot(delta[k][:,0],delta[k][:,0]) for k in active_interaction_set])+\
        lam[1]*(np.count_nonzero(zeta[0,:]))+\
        r*lam[1]*(np.count_nonzero(alpha[0,:]))
    
    J_initial = deepcopy(J)
    active_set_update = np.array([x for x in active_set if x not in np.where(zeta[0,:] == 1)[0]])
    active_interaction_set_update = np.array([x for x in active_interaction_set if x not in np.where(alpha[0,:] == 1)[0]])
    if verbose == True:
        display(Math(r'Input~Obj: {:.0f},'.format(J)+'\sum_{j \in S^c} z_j: '+'{} \leq {}.'.format(np.count_nonzero(zeta[0,:]), len(active_set))+'\sum_{ij \in S^c} z_{ij}: '+'{} \leq {}.'.format(np.count_nonzero(alpha[0,:]),len(active_interaction_set)))) 
    
    for it in range(max_iter):
        for j in active_set:
            if zeta[0,j]==True:
                Ypred -= Bspam[j].dot(beta[j])
            res = Y-Ypred
            beta[j], zeta[:,j] = utilities.solve(B = Bspam[j], P = Pspam[j], y = res, beta = beta[j], S=S[j], lam = [lam[0], lam[1]])
            if zeta[0,j]==True:
                Ypred += Bspam[j].dot(beta[j])
        for j in active_interaction_set:
            if alpha[0,j]==True:
                Ypred -= Bspam_interaction[j].dot(delta[j])
            res = Y-Ypred
            delta[j], alpha[:,j] = utilities.solve(B = Bspam_interaction[j], P = Pspam_interaction[j], y = res, beta = delta[j], S=S_interaction[j], lam = [lam[0], r*lam[1]])
            if alpha[0,j]==True:
                Ypred += Bspam_interaction[j].dot(delta[j])
        
        J_prev = deepcopy(J)
        J = 0.5*mean_squared_error(Y, Ypred)+\
            lam[0]*sum([(np.transpose(beta[k])).dot(S[k].dot(beta[k]))[0,0] for k in active_set])+\
            lam[0]*sum([(np.transpose(delta[k])).dot(S_interaction[k].dot(delta[k]))[0,0] for k in active_interaction_set])+\
            eps*sum([np.dot(beta[k][:,0],beta[k][:,0]) for k in active_set])+\
            eps*sum([np.dot(delta[k][:,0],delta[k][:,0]) for k in active_interaction_set])+\
            lam[1]*(np.count_nonzero(zeta[0,:]))+\
            r*lam[1]*(np.count_nonzero(alpha[0,:]))
        if J>10*J_initial:
            beta = [np.zeros(bb.shape,dtype=float) for bb in beta]
            delta = [np.zeros(dd.shape,dtype=float) for dd in delta]
            Ypred = np.mean(Y)*np.ones(Y.shape,dtype=float)
            J_initial = 0.5*mean_squared_error(Y, Ypred)+\
                        lam[0]*sum([(np.transpose(beta[k])).dot(S[k].dot(beta[k]))[0,0] for k in active_set])+\
                        lam[0]*sum([(np.transpose(delta[k])).dot(S_interaction[k].dot(delta[k]))[0,0] for k in active_interaction_set])+\
                        eps*sum([np.dot(beta[k][:,0],beta[k][:,0]) for k in active_set])+\
                        eps*sum([np.dot(delta[k][:,0],delta[k][:,0]) for k in active_interaction_set])+\
                        lam[1]*(np.count_nonzero(zeta[0,:]))+\
                        r*lam[1]*(np.count_nonzero(alpha[0,:]))
            debugging_mode = True
#             for i in active_set_update:
#                 Pspam[i] = sp.linalg.splu((Bspam[i].transpose()).dot(Bspam[i])+2*N*(lam[0]*S[i]+eps*sp.csr_matrix(np.identity(Bspam[i].shape[1]))))
#             for i in active_interaction_set_update:
#                 Pspam_interaction[i] = sp.linalg.splu((Bspam_interaction[i].transpose()).dot(Bspam_interaction[i])+2*N*(lam[0]*S_interaction[i]+eps*sp.csr_matrix(np.identity(Bspam_interaction[i].shape[1]))))
            continue
    
        J_del = J-J_prev
        if debugging_mode == True:
            print('Debugging, lambda_1:{:.6f}, lambda_2:{:.6f}, J: {:.5f}, |Delta J/J|: {:.4f}, '.format(lam[0], lam[1], J, np.absolute(J_del/J)))
        if np.absolute(J_del/J)<tol:
            break
    
        
    if verbose == True:
        display(Math(r'Output~Obj: {:.0f}, |\Delta J/J|: {:.4f}, '.format(J, np.absolute(J_del/J))+'\sum_{j \in S^c} z_j: '+'{} \leq {}.'.format(np.count_nonzero(zeta[0,:]), len(active_set))+'\sum_{ij \in S^c} z_{ij}: '+'{} \leq {}.'.format(np.count_nonzero(alpha[0,:]),len(active_interaction_set)))) 
    if(it == max_iter-1):
        with open(path+'/Warning.txt', "a") as f:
            f.write('Warning: CD over active set did not converge within the chosen max_iter!')
            
    return Ypred, beta, zeta, delta, alpha

def CD_Joint(CD_J_AS = None,
             Ypred = None,
             beta = None,
             zeta = None,
             active_set = None,
             lam = None,
             P = None,
             P_interaction = None,
             Y = None,
             B = None,
             B_interaction = None,
             S = None,
             S_interaction = None,
             I = None,
             interaction_terms = None,
             r = None,
             max_iter = None,
             tol = 1e-4,
             full_set = None,
             MaxSuppSize_main = None,
             MaxSuppSize_interaction = None,
             verbose = False,
             path = None):
    """Cyclic Block Coordinate Descent over the full set of main/interaction effects.
    
    Args:
        CD_J_AS: a callable function that optimizes over a reduced set of main effects, callable.
        Ypred: numpy array of shape (N, ).
        beta: coefficients for main/interaction effects, 2 lists of arrays of shapes [ [(Ki+1, 1), ...], [(Kij+1, 1), ...]]
        zeta: binary vector to track which main effects are in the active set, 2 bool arrays of shape [(1, d), (1, Imax)]
        active_set: indices of main effects to optimize over, a numpy int array.
        lam: regularization parameters [lam_1, lam_2], list of floats.
        P: B^T*B + 2*N*(lam_1*S_i + eps*I) matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
            eps is a small epsilon for numerical stability.
        P_interaction: B^T*B + 2*N*(lam_1*S_ij + eps*I) matrices for main effects, list of sparse matrices of shapes [(Kij+1, Kij+1), ...].
            eps is a small epsilon for numerical stability.
        Y: training target responses, a float numpy array of shape (N,).
        B: B-spline transformed sparse matrices for main effects, list of sparse matrices of shapes [(N, Ki+1), ...].
        B_interaction: B-spline transformed sparse matrices for interaction effects, list of sparse matrices of shapes [(N, Kij+1), ...].
        S: Smoothness matrices for main effects, list of sparse matrices of shapes [(Ki+1, Ki+1), ...].
        S_interaction: Smoothness matrices for interaction effects, list of sparse matrices of shapes [(Kij+1, Kij+1), ...].
        I: number of possible main/interaction effects, int scalers.
        interaction_terms: list of interaction effects to consider if only a subset need to be considered, 
            a 2D numpy array of of shape (Imax, 2).
        r: relative scaling factor for L0 penalty between main and interaction effects.
            We consider r=1.0 (corresponds to alpha symbol in the paper), float scaler. 
        max_iter: maximum number of Cyclic BCD on the active set, int scaler.
        tol: relative loss termination criteria for stopping, a float scalar.
        full_set: indices of all main effects, a numpy int array.
        main_terms: list of main effects to consider if only a subset need to be considered, 
            not supported yet.
        MaxSuppSize_main: Stop L0 regularization if the active set of main effects is larger than the MaxSuppSize_main
            and move to next smoothing lambda setting and start L0 regularization, int scaler.
        MaxSuppSize_interaction: Stop L0 regularization if the active set of interaction effects is larger than the MaxSuppSize_interaction
            and move to next smoothing lambda setting and start L0 regularization, int scaler.
        verbose: for printing optimization steps, bool scaler.
        path: for logging, str.
    
    Returns:
        Ypred: Updated prediction, numpy array of shape (N, ).
        beta: Updated coefficients for main effects, list of arrays of shapes [(Ki+1, 1), ...].
        zeta: Updated binary vector to track which main effects are in the active set, a bool array of shape (1, d).
        delta: Updated coefficients for interaction effects, list of arrays of shapes [(Kij+1, 1), ...].
        alpha: Updated binary vector to track which interaction effects are in the active set, a bool array of shape (1, Imax).
        active_set: Updated indices of nonzero main effects, a numpy int array.
        active_interaction_set: Updated indices of nonzero interaction effects, a numpy int array.
        MaxSuppSize_flag: indicates Maximum Support size is reached, bool scaler.
    """
    N = Y.shape[0]
    delta = beta[1]
    beta = beta[0]
    alpha = zeta[1]
    zeta = zeta[0]
    active_interaction_set = active_set[1]
    active_set = active_set[0]
    full_interaction_set = full_set[1]
    full_set = full_set[0] 
    Bspam = B
    Bspam_interaction = B_interaction
    Pspam = P
    Pspam_interaction = P_interaction
    d = I[0]
    dinteraction = I[1]
    MaxSuppSize_flag = 0
    eps = 1e-8
    
    warnings.filterwarnings("error")
    res = Y-Ypred
    beta_p = [(P.solve((B.transpose()).dot(res))).reshape(-1,1) for B, P in zip(Bspam, Pspam)]
    res_p = np.array([np.linalg.norm(res-B.dot(bp)) for B, bp in zip(Bspam, beta_p)])
    active_set = np.arange(d)
#     if active_set is None: 
#         A = int(np.ceil(0.1*d))
#         active_set = res_p.argsort()[:A]        
#     else:
#         A = np.minimum(np.maximum(int(np.ceil(0.2*len(active_set))),10), 50)
#         active_set = np.union1d(active_set, res_p.argsort()[:A]) 
                
    res = Y-Ypred 
    delta_p = [(P.solve((B.transpose()).dot(res))).reshape(-1,1) for B, P in zip(Bspam_interaction, Pspam_interaction)]
    res_p = np.array([np.linalg.norm(res-B.dot(dp)) for B, dp in zip(Bspam_interaction, delta_p)])        
    if active_interaction_set is None: 
        A = int(np.ceil(0.01*dinteraction))
        active_interaction_set = res_p.argsort()[:A]
    else:    
        A = np.minimum(np.maximum(int(np.ceil(0.2*len(active_interaction_set))),10), 50)
        active_interaction_set = np.union1d(active_interaction_set, res_p.argsort()[:A]) 

    '''
        Coordinate Descent over full set
    '''

    for it in range(max_iter):
        Ypred, beta, zeta, delta, alpha = CD_J_AS(Ypred = Ypred,
                                                  beta = [beta, delta],
                                                  zeta = [zeta, alpha],
                                                  active_set = [active_set, active_interaction_set],
                                                  lam = [lam[0], lam[1]],
                                                  P = Pspam,
                                                  P_interaction = Pspam_interaction)
        
        active_set = np.where(zeta[0,:] == 1)[0]
        active_interaction_set = np.where(alpha[0,:] == 1)[0]
        if (len(np.where(zeta[0,:] == 1)[0]) > MaxSuppSize_main) or (len(np.where(alpha[0,:] == 1)[0]) > MaxSuppSize_interaction):
            MaxSuppSize_flag = 1
            break
        J = 0.5*mean_squared_error(Y, Ypred)+\
            lam[0]*sum([(np.transpose(beta[k])).dot(S[k].dot(beta[k]))[0,0] for k in active_set])+\
            lam[0]*sum([(np.transpose(delta[k])).dot(S_interaction[k].dot(delta[k]))[0,0] for k in active_interaction_set])+\
            eps*sum([np.dot(beta[k][:,0],beta[k][:,0]) for k in active_set])+\
            eps*sum([np.dot(delta[k][:,0],delta[k][:,0]) for k in active_interaction_set])+\
            lam[1]*(np.count_nonzero(zeta[0,:]))+\
            r*lam[1]*(np.count_nonzero(alpha[0,:]))    

        if verbose == True:
            display(Math(r'Iteration: {}, Obj: {:.0f}, '.format(it, J)+', \sum_{j \in S^c} z_j: '+'{} \leq {}.'.format(np.count_nonzero(zeta[0,:]), len(active_set))+'\sum_{ij \in S^c} z_{ij}: '+'{} \leq {}.'.format(np.count_nonzero(alpha[0,:]),len(active_interaction_set)))) 

        for j in [x for x in full_set if x not in active_set]:
            if zeta[0,j]==1:
                Ypred -= Bspam[j].dot(beta[j])
            res = Y-Ypred
            beta[j], zeta[:,j] = utilities.solve(B=Bspam[j], P=Pspam[j], y=res, beta=beta[j], S=S[j], lam=[lam[0], lam[1]])
            if zeta[0,j]==1:
                Ypred += Bspam[j].dot(beta[j])
        for j in [x for x in full_interaction_set if x not in active_interaction_set]:
            if alpha[0,j]==1:
                Ypred -= Bspam_interaction[j].dot(delta[j])
            res = Y-Ypred
            delta[j], alpha[:,j] = utilities.solve(B=Bspam_interaction[j], P=Pspam_interaction[j], y=res, beta=delta[j], S=S_interaction[j], lam=[lam[0], r*lam[1]])
            if alpha[0,j]==1:
                Ypred += Bspam_interaction[j].dot(delta[j])
        if np.count_nonzero(zeta[0,:])==active_set.shape[0] and np.count_nonzero(alpha[0,:])==active_interaction_set.shape[0]:
            if np.sum(sorted(active_set) == np.where(zeta[0,:] == 1)[0])==active_set.shape[0] and np.sum(sorted(active_interaction_set) == np.where(alpha[0,:] == 1)[0])==active_interaction_set.shape[0]:
                #print('Active set converged')
                active_set = np.where(zeta[0,:] == 1)[0]
                active_interaction_set = np.where(alpha[0,:] == 1)[0]
                break
        active_set = np.where(zeta[0,:] == 1)[0]
        active_interaction_set = np.where(alpha[0,:] == 1)[0]
#     for i in active_set:
#         Pspam[i] = sp.linalg.splu((Bspam[i].transpose()).dot(Bspam[i])+2*N*(lam[0]*S[i]+eps*sp.csr_matrix(np.identity(Bspam[i].shape[1]))))
#     for i in active_interaction_set:
#         Pspam_interaction[i] = sp.linalg.splu((Bspam_interaction[i].transpose()).dot(Bspam_interaction[i])+2*N*(lam[0]*S_interaction[i]+eps*sp.csr_matrix(np.identity(Bspam_interaction[i].shape[1]))))

    if(it == max_iter-1):
        with open(path+'/Warning.txt', "a") as f:
            f.write('Warning: CD over full set did not converge within the chosen max_iter!')
            f.write('\lambda_1: {:.7f},\lambda_2: {:.7f}'.format(lam[0], lam[1]))

    return Ypred, beta, zeta, delta, alpha, active_set, active_interaction_set, MaxSuppSize_flag