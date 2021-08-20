from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB
from IPython.display import display
from IPython.display import Math
from ipywidgets import *
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.linalg import block_diag
from scipy.special import comb
from sklearn.metrics import mean_squared_error
from tqdm import tnrange, tqdm_notebook

def MIP_HierarchicalSparsity_ActiveSet(Ypred = None,
                                       beta = None,
                                       zeta = None,
                                       lam = None,
                                       active_set = None,
                                       Y = None,
                                       B = None,
                                       S = None,
                                       I = None,
                                       interaction_terms = None,
                                       r = None,
                                       verbose=False,
                                       path = None,
                                       threads = 10,
                                       time_limit = 6):
    """Solves convex relaxation of the MIP formulation for nonparametric additive models with interactions under strong hierarchy.
    
    Args:
        Ypred: current prediction, numpy array of shape (N, ).
        beta: coefficients for main/interaction effects, 2 lists of arrays of shapes [ [(Ki+1, 1), ...], [(Kij+1, 1), ...]]
        zeta: binary vector to track which main effects are in the active set, 2 bool arrays of shape [(1, d), (1, Imax)]
        lam: regularization parameters [lam_1, lam_2], list of floats.
        active_set: indices of main effects to optimize over, a numpy int array.
        Y: training target responses, a float numpy array of shape (N,).
        B: B-spline transformed sparse matrices for main/interaction effects, 2 lists of sparse matrices of shapes [[(N, Ki+1), ...], [(N, Kij+1), ...]].
        S: Smoothness matrices for main/interaction effects, 2 lists of sparse matrices of shapes [[(Ki+1, Ki+1), ...], [(Kij+1, Kij+1), ...]].
        I: number of maximum main/interaction effects, shape (2,).
        interaction_terms: list of interaction effects to consider if only a subset need to be considered, 
            a 2D numpy array of of shape (Imax, 2).
        r: relative scaling factor for L0 penalty between main and interaction effects.
            We consider r=1.0 (corresponds to alpha symbol in the paper), float scaler. 
        verbose: whether to print optimization log from gurobi, bool scaler.
        path: folder path to log results to, str.
        threads: number of threads in parallel used by Gurobi, int scaler.
        time_limit: number of maximum hours used by Gurobi, int scaler.   
        
    Returns:
        Ypred: updated prediction, numpy array of shape (N, ).
        beta: coefficients for main effects, list of arrays of shapes [(Ki+1, 1), ...].
        zeta: binary vector to track which main effects are in the active set, a bool array of shape (1, d)
            corresponds to z_i's in the paper.
        delta: coefficients for interaction effects, list of arrays of shapes [(Kij+1, 1), ...].
            corresponds to theta in the paper.
        alpha: binary vector to track which interactions effects are in the active interaction set, a bool array of shape (1, Imax)
            corresponds to z_ij's in the paper.
    
    """

    Ypred = np.mean(Y)*np.ones(Y.shape,dtype=float)
    M = 1e3
    eps = 1e-8
    N = Y.shape[0]
    delta = deepcopy(beta[1])
    beta = deepcopy(beta[0])
    alpha = deepcopy(zeta[1])
    zeta = deepcopy(zeta[0])

    active_interaction_set = deepcopy(active_set[1])
    active_set = deepcopy(active_set[0])
    active_set = np.sort(np.union1d(active_set, np.unique(np.array([interaction_terms[k] for k in active_interaction_set]))))
    Bspam = B[0]
    Bspam_interaction = B[1]
    S_interaction = S[1]
    S = S[0]
    
    beta_AS = deepcopy([beta[j] for j in active_set])
    delta_AS = deepcopy([delta[j] for j in active_interaction_set])
    
    Kn_AS = [Bspam[j].shape[1] for j in active_set]
    Kn_interaction_AS = [Bspam_interaction[j].shape[1] for j in active_interaction_set]
    Bspam_AS = [Bspam[j] for j in active_set]
    Bspam_interaction_AS = [Bspam_interaction[j] for j in active_interaction_set]
    K_c = np.array(np.cumsum(Kn_AS))
    K_interaction_c = np.array(np.cumsum(Kn_interaction_AS))                        

    S_AS = [S[j] for j in active_set] 
    S_interaction_AS = [S_interaction[j] for j in active_interaction_set]                        

    P_AS = sp.hstack(Bspam_AS)
    P_interaction_AS = sp.hstack(Bspam_interaction_AS)   
    Q_AS = sp.block_diag(S_AS)
    Q_interaction_AS = sp.block_diag(S_interaction_AS)    
    P = sp.hstack([P_AS, P_interaction_AS]).toarray()
    Q = sp.block_diag([Q_AS, Q_interaction_AS]).toarray()

    
    interaction_terms_AS = [interaction_terms[j] for j in active_interaction_set]
    
    coupled_terms = []                        
    for index_j, j in enumerate(active_set):
        terms = []                    
        for index_ij, (f_i, f_j) in enumerate(interaction_terms_AS):
            if j==f_i:
                terms.append((index_j, index_ij))
            elif j==f_j:
                terms.append((index_j, index_ij)) 
        if len(terms)>0:                   
            coupled_terms.append(terms)                 
    coupled_terms = np.concatenate(coupled_terms) 
                            
    # Gurobi Optimization Algorithm for Hierarchical Sparsity
    # Initialize Gurobi Model    
    mod = gp.Model("HS")
    #     mod.setParam('OutputFlag', False)
    #     mod.setParam('Threads',threads)
    #     mod.setParam('TimeLimit', int(time_limit*3600))
    #     mod.setParam('MIQCPMethod',0)
    mod.setParam('LogFile', path+'/gurobi.log')

    # Build variables
    omega_var = mod.addMVar(shape=sum(Kn_AS)+sum(Kn_interaction_AS), vtype=GRB.CONTINUOUS, lb =  -GRB.INFINITY, ub = GRB.INFINITY, name="beta")
    zeta_var = mod.addMVar(shape=len(active_set), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="zeta")   
    alpha_var = mod.addMVar(shape=len(active_interaction_set), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="alpha")  


    # Build Constraints
    # Binary <-> Continuous constraints
    if len(active_set)>0:                    
        mod.addQConstr((omega_var[:K_c[0]]@omega_var[:K_c[0]])<=M*(zeta_var[0]))
        if len(active_set)>1:                       
            [mod.addQConstr((omega_var[K_c[j]:K_c[j+1]]@omega_var[K_c[j]:K_c[j+1]])<=M*(zeta_var[j+1])) for j in range(len(active_set)-1)]
    if len(active_interaction_set)>0:                                
        mod.addQConstr((omega_var[K_c[-1]:(K_c[-1]+K_interaction_c[0])]@omega_var[K_c[-1]:(K_c[-1]+K_interaction_c[0])])<=M*(alpha_var[0]))
        if len(active_interaction_set)>1:                                
            [mod.addQConstr((omega_var[(K_c[-1]+K_interaction_c[j]):(K_c[-1]+K_interaction_c[j+1])]@omega_var[(K_c[-1]+K_interaction_c[j]):(K_c[-1]+K_interaction_c[j+1])])<=M*(alpha_var[j+1])) for j in range(len(active_interaction_set)-1)]
    # Binary constraints
    if len(active_interaction_set)>0: 
        [mod.addConstr(alpha_var[term[1]]<=zeta_var[term[0]]) for term in coupled_terms]

    # update model with constraints
    mod.update()                            

    # Build Objective
    res = Y-Ypred
    YTX = np.matmul(np.transpose(res), P)[0,:]
    XTX = np.matmul(np.transpose(P), P)

    mod.setObjective((1/(2*N))*(np.dot(res[:,0], res[:,0])\
                                -2*(YTX@omega_var)\
                                +omega_var@(XTX+(2*N)*(lam[0]*Q + eps*np.identity(omega_var.shape[0])))@omega_var)\
                     +lam[1]*(np.ones(zeta_var.shape[0])@zeta_var)\
                     +r*lam[1]*(np.ones(alpha_var.shape[0])@alpha_var)\
                     )
                     
    
    # Update model with objective
    mod.update()

    # Execute optimization algorithm
    mod.optimize()
        
    omega = np.zeros((XTX.shape[0],1),dtype=float)
    for j in range(sum(Kn_AS)+sum(Kn_interaction_AS)):
        omega[j,0] = omega_var[j].X
    beta_hat = omega[:K_c[-1]].reshape(-1,1)
    delta_hat = omega[K_c[-1]:].reshape(-1,1)        
    
    # Extract beta and theta values
    zeta_hat = np.zeros((len(active_set),),dtype=float)                             
    alpha_hat = np.zeros((len(active_interaction_set),),dtype=float)                             
    for j in range(zeta_hat.shape[0]):
        zeta_hat[j] = zeta_var[j].X
    for j in range(alpha_hat.shape[0]):
        alpha_hat[j] = alpha_var[j].X

    mod.write(path+"/model.sol")
        

    beta_AS[0] = beta_hat[np.arange(0,K_c[0]),0].reshape(-1,1)
    for j in range(len(active_set)-1):
        beta_AS[j+1] = beta_hat[np.arange(K_c[j],K_c[j+1]),0].reshape(-1,1)                         

    delta_AS[0] = delta_hat[np.arange(0,K_interaction_c[0]),0].reshape(-1,1)
    for j in range(len(active_interaction_set)-1):
        delta_AS[j+1] = delta_hat[np.arange(K_interaction_c[j],K_interaction_c[j+1]),0].reshape(-1,1)                         
        
    for j, item in enumerate(active_set):
        beta[item] = deepcopy(beta_AS[j])
        zeta[0,item] = deepcopy(zeta_hat[j])                         
    for j, item in enumerate(active_interaction_set):
        delta[item] = deepcopy(delta_AS[j])
        alpha[0,item] = deepcopy(alpha_hat[j])                          
                                     
    Ypred = np.mean(Y)*np.ones(Y.shape,dtype=float)\
            +np.array(sum([B.dot(b) for b, B in zip(beta_AS, Bspam_AS)])).reshape(Y.shape)\
            +np.array(sum([B.dot(d) for d, B in zip(delta_AS, Bspam_interaction_AS)])).reshape(Y.shape)        
    J = 0.5*mean_squared_error(Y, Ypred)+\
        lam[0]*sum([(np.transpose(beta[k])).dot(S[k].dot(beta[k]))[0,0] for k in active_set])+\
        lam[0]*sum([(np.transpose(delta[k])).dot(S_interaction[k].dot(delta[k]))[0,0] for k in active_interaction_set])+\
        lam[1]*(np.sum(zeta[0,:]))+\
        r*lam[1]*(np.sum(alpha[0,:]))    
    if verbose == True:
        display(Math(r'Output~Obj: {:.6f}, '.format(J)+'\sum_{j \in S^c} z_j: '+'{}.'.format(np.count_nonzero(zeta[0,:]))+'\sum_{ij \in S^c} z_{ij}: '+'{} \leq {}.'.format(np.count_nonzero(alpha[0,:]),len(active_interaction_set)))) 
    active_set = np.where(zeta[0,:] == 1)[0]
    active_interaction_set = np.where(alpha[0,:] == 1)[0]
                                     
    return Ypred, deepcopy(beta), deepcopy(zeta), deepcopy(delta), deepcopy(alpha)