from __future__ import division, print_function
from contextlib import redirect_stdout
from copy import deepcopy
from functools import partial
from IPython.display import Math
from ipywidgets import *
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from patsy import dmatrix
import scipy.sparse as sp
from scipy.special import comb
from sklearn import metrics
import sys
import time
from tqdm import notebook
import warnings

from SparseAMsWithInteractions.src.AMsWithInteractionsL0 import utilities
from SparseAMsWithInteractions.src.AMsWithInteractionsL0 import CrossValidation
from SparseAMsWithInteractions.src.AMsWithInteractionsL0 import L0Path
from SparseAMsWithInteractions.src.AMsWithInteractionsL0 import optimizer
from SparseAMsWithInteractions.src import utils

os.environ['QT_QPA_PLATFORM']='offscreen'
font = {'weight' : 'bold',
        'size'   : 14}

class AMI(object):
    """AM with interactions (AMI) with b-splines under L0 sparsity.
    
    Attributes:
        lams_sm: Regularization path over smoothness penalty for spline bases, float numpy 1D array. 
        lams_L0: Regularization path for L0 penalty for sparsity, float numpy 1D array.
        alpha: relative L0 penalty for interaction effects compared to main effects.
        max_interaction_support: Maximum interaction support at which the regularization path is terminated, scalar int.
        convergence_tolerance: relative loss termination criteria for stopping, a float scalar.
        max_iter: maximum number of iterations for partially greedy Cyclic Coordinate Descent, int scalar
        eps: small epsilon added to QP for numerical stability for active set screening/updates, a float scalar.
        val_criteria: evaluation metric for hyperparameter tuning,
          - 'mse'
          - 'mae'
        val_crit_opt: optimal evaluation metric achieved.
        val_crit_sp: sparse evaluation metric achieved within 1% of optimal solution along L0 regularization path.
        
        X: training covariates, a float numpy array of shape (N, p).
        Y: training target responses, a float numpy array of shape (N,).
        Xval: validation covariates, a float numpy array of shape (Nval, p).
        Yval: validation target responses, a float numpy array of shape (Nval,).

        p: number of covariates, int scalar.
    """
    def __init__(self,
                 lams_sm=np.logspace(start=-3, stop=-7, num=20, base=10.0),
                 lams_L0=np.logspace(start=0, stop=-4, num=25, base=10.0),
                 alpha=1.0,
                 max_interaction_support=1000,
                 convergence_tolerance=1e-4,
                 max_iter=1000,
                 eps=1e-8,
                 eval_criteria='mse',
                 degree=3,
                 path=None,
                 logging=True,
                 terminate_val_L0path=True):
        assert path is not None
        os.makedirs(path, exist_ok=True)
        
        self.lams_sm = np.sort(lams_sm)[::-1]
        self.lams_L0 = np.sort(lams_L0)[::-1]
        self.alpha = alpha
        self.max_interaction_support = max_interaction_support
        self.convergence_tolerance = convergence_tolerance
        self.max_iter = max_iter
        self.eps = eps
        if eval_criteria in ['mse', 'mae']:
            self.eval_criteria = eval_criteria
        else:
            raise ValueError("Evaluation criteria {} is not supported".format(eval_criteria))
        self.path = path
        self.degree = degree
        self.logging = logging
        self.terminate_val_L0path = terminate_val_L0path
    
    def load_data(self, X, Y, y_scaler, column_names, Xmin, Xmax, eps=1e-6):
        self.X = X
        self.Y = Y
        self.y_scaler = y_scaler
        self.column_names = column_names
        self.N, self.p = self.X.shape
        # eps added to give a slight margin at boundaries for spline generation
        self.Xmin = Xmin - eps 
        self.Xmax = Xmax + eps        
        
    def generate_interaction_terms(self, generate_all_pairs=True, Imax=10000, subsample=0.2, Kij=4):
        """Generates pairwise indices of interaction effects.
        
        Either generates all possible pairwise or maximum screened top I pairs
        Args:
            generate_all_pairs: whether to use generate interaction pairs, boolean.
            Imax: maximum number of interactions to be considered based on marginal fits, int scaler.
                ignored when generate_all_pairs=True.
            subsample: number of samples to consider for marginal fits, float scaler,
                ignored when generate_all_pairs=True.
            Kij: degrees of freedom for interaction terms, int scaler,
                ignored when generate_all_pairs=True.
        """
        self.generate_all_pairs = generate_all_pairs
        self.interaction_terms = []
        for m in range(0, self.p):
            for n in range(0, self.p):
                if m!=n and m<n:
                    self.interaction_terms.append((m, n))
        self.interaction_terms = np.array(self.interaction_terms)
        self.I = (int)(comb(self.p, 2, exact=False))
        assert self.I==len(self.interaction_terms), "Number of total interactions do not match!"
        self.Imax = self.I
        
        if not self.generate_all_pairs:
            assert Imax <= self.I, "Imax:{} must be less than maximum number of interactions:{}".format(Imax, self.I)
            self.Imax = Imax
            num_cores = mp.cpu_count() 
            batches = num_cores
            batch_size = int(np.floor(self.I/batches))
            interaction_terms_batches = []
            for i in range(batches-1):
                interaction_terms_batches.append(self.interaction_terms[int(i*batch_size):int((i+1)*batch_size)])
            interaction_terms_batches.append(self.interaction_terms[int((batches-1)*batch_size):])

            idx = np.random.randint(self.N, size=int(np.ceil(subsample*self.N)))
            func = partial(utilities.screening, X=self.X[idx], Y=self.Y[idx], Xmin=self.Xmin, Xmax=self.Xmax, Kij=Kij) 

            with mp.Pool(num_cores) as pool:
                results = list(notebook.tqdm(pool.imap(func, interaction_terms_batches), total=len(interaction_terms_batches)))
                pool.close()
                pool.join()

            res_p = np.argsort([item for sublist in results for item in sublist])[:self.Imax]
            self.interaction_terms = np.array([self.interaction_terms[k] for k in res_p])            
            
        
    def generate_splines_and_quadratic_penalties(self, Ki, Kij):
        """Generates b-splines and quadratic penalties and reduced BTB matrices.
        
        Ki: Degrees of freedom for b-spline basis, int scalar.
        Kij: Degrees of freedom for b-spline basis in each covariate direction, int scalar.        
        """
        self.Ki = Ki
        self.Kij = Kij
        self.Btrain, self.Btrain_interaction, self.K_main, self.K_interaction = utilities.generate_bspline_transformed_X(self.X, self.Xmin, self.Xmax, self.Ki, self.Kij, self.interaction_terms)
        self.S, self.S_interaction = utilities.generate_bspline_quadratic_penalties(self.K_main, self.K_interaction, self.interaction_terms)
        self.BtrainT_Btrain = [(B.transpose()).dot(B) for B in self.Btrain]
        self.Btrain_interactionT_Btrain_interaction = [(B.transpose()).dot(B) for B in self.Btrain_interaction]
        
    def fitCV(self, Xval, Yval):
        self.Xval = Xval
        self.Yval = Yval
        self.CD_J_AS = (lambda Ypred, beta, zeta, active_set, lam, P, P_interaction : optimizer.CD_Joint_ActiveSet(
            Ypred, beta, zeta, active_set, lam, P, P_interaction,
            Y=self.Y, B=self.Btrain, B_interaction=self.Btrain_interaction, S=self.S, S_interaction=self.S_interaction, I=[self.p, len(self.interaction_terms)], interaction_terms=self.interaction_terms, r=self.alpha, max_iter=self.max_iter, tol=self.convergence_tolerance, verbose=False, path=self.path)
        )
        self.CD_J = (lambda CD_J_AS, Ypred, beta, zeta, active_set, lam, P, P_interaction : optimizer.CD_Joint(
            CD_J_AS, Ypred, beta, zeta, active_set, lam, P, P_interaction,
            Y=self.Y, B=self.Btrain, B_interaction=self.Btrain_interaction, S=self.S, S_interaction=self.S_interaction, I=[self.p, len(self.interaction_terms)], interaction_terms=self.interaction_terms, r=self.alpha, max_iter=100, tol=self.convergence_tolerance, full_set=[np.arange(self.p), np.arange(len(self.interaction_terms))], MaxSuppSize_main=self.p, MaxSuppSize_interaction=self.max_interaction_support, path=self.path)
        )
        
        self.L0path = (lambda CD_J, CD_J_AS, lam_1, lams_2, active_set, active_interaction_set, beta, zeta, delta, alpha :L0Path.L0Path(
            CD_J, CD_J_AS, lam_1, lams_2, active_set, active_interaction_set, beta, zeta, delta, alpha,
            B=self.Btrain, BT_B=self.BtrainT_Btrain, B_interaction=self.Btrain_interaction, B_interactionT_B_interaction=self.Btrain_interactionT_Btrain_interaction, K_main=self.K_main, K_interaction=self.K_interaction, Xval=self.Xval, Xmin=self.Xmin, Xmax=self.Xmax, Y=self.Y, Yval=self.Yval, y_scaler=self.y_scaler, S=self.S, S_interaction=self.S_interaction, interaction_terms=self.interaction_terms, eval_criteria=self.eval_criteria, path=self.path, r=self.alpha, logging=self.logging, terminate_val_L0path=self.terminate_val_L0path)
        )
        
        self.CV = (lambda L0path, CD_J, CD_J_AS: CrossValidation.CrossValidation(
            L0path, CD_J, CD_J_AS,
            B=self.Btrain, B_interaction=self.Btrain_interaction, interaction_terms=self.interaction_terms, column_names=self.column_names, lams_1=self.lams_sm, lams_2=self.lams_L0, path=self.path, logging=self.logging)
        )
        
        start = time.time()
        self.optimal_solution, self.sparse_solution, self.union_set = self.CV(L0path=self.L0path, CD_J=self.CD_J, CD_J_AS=self.CD_J_AS)
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        if self.logging == True:
            with open(self.path+'/Results.txt', "a") as f:
                f.write("Training completed in {:0>2}:{:0>2}:{:05.2f} \n".format(int(hours),int(minutes),seconds))
        print("Training completed in {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        self.active_set_union = self.union_set[0]
        self.active_interaction_set_union = self.union_set[1]
        self.interaction_terms_union = np.array([self.interaction_terms[k] for k in self.active_interaction_set_union])
        self.beta_opt, self.delta_opt, self.zeta_opt, self.alpha_opt, self.lam_sm_opt, self.lam_L0_opt, self.J_opt, self.active_set_opt, self.active_interaction_set_opt = self.optimal_solution
        self.beta_sp, self.delta_sp, self.zeta_sp, self.alpha_sp, self.lam_sm_sp, self.lam_L0_sp, self.J_sp, self.active_set_sp, self.active_interaction_set_sp = self.sparse_solution
    
    def generate_splines_on_active_set(self, X, active_set, active_interaction_set):
        """Generates splines on active set.
        
        Args:
            X: test data matrix, numpy array of shape (N, p).
            active_set: main effects to consider, int numpy array of shape (p, ).
        
        Returns:
            B: B-spline transformed matrices on active set, list of sparse matrices of shapes [(Ki+1,), ...].
            B_interaction: B-spline transformed matrices on active interaction set,
                list of sparse matrices of shapes [(Ki*Ki+1,), ...].
        """
        # Generate b-splines on active set
        B = [None]*self.p
        for k in active_set:
            B[k] = sp.csr_matrix(
                np.array(
                    dmatrix("bs(x, df={}, degree={}, include_intercept=False, lower_bound={}, upper_bound={})".format(
                        self.K_main[k], self.degree, self.Xmin[k], self.Xmax[k]), {"x": X[:, k]}
                    )
                ), dtype=np.float64)
           
        B_interaction = [None]*len(self.interaction_terms)
        for k in active_interaction_set:
            f_i, f_j = self.interaction_terms[k]
            B_interaction[k] = sp.csr_matrix(
                np.array(
                    dmatrix("te(bs(x1, df={}, degree={}, include_intercept=False, lower_bound={}, upper_bound={}), bs(x2, df={}, degree={}, include_intercept=False, lower_bound={}, upper_bound={}))".format(
                        self.K_interaction[f_i], self.degree, self.Xmin[f_i], self.Xmax[f_i], self.K_interaction[f_j], self.degree, self.Xmin[f_j], self.Xmax[f_j]), {"x1": X[:, f_i], "x2": X[:, f_j]}
                    )
                ), dtype=np.float64)
        
        return B, B_interaction 

    def predict(self, X, use_sparse_solution=False): 
        """Generates spline transformations on new data and predicts the response.
        
        Args:
            X: test data matrix, numpy array of shape (N, p).
            use_sparse_solution: whether to use optimal solution or sparse solution, bool scalar.
        
        Returns:
            Ytespred: numpy array of shape (N, ).
        """
        # Clip to handle covariate instances that maybe outside the spline basis generation
        for i in range(self.p):
            X[:,i] = np.clip(X[:,i], a_min=self.Xmin[i], a_max=self.Xmax[i]) 

        if use_sparse_solution:
            beta = self.beta_sp
            delta = self.delta_sp
            active_set = self.active_set_sp
            active_interaction_set = self.active_interaction_set_sp
        else:
            beta = self.beta_opt
            delta = self.delta_opt
            active_set = self.active_set_opt
            active_interaction_set = self.active_interaction_set_opt

        # Generate b-splines on active set
        B, B_interaction = self.generate_splines_on_active_set(X, active_set, active_interaction_set)

        # Prediction
        Ypred = np.mean(self.Y) \
                + np.array(sum([B[j].dot(beta[j]) for j in active_set]))\
                + np.array(sum([B_interaction[j].dot(delta[j]) for j in active_interaction_set]))
        
        return Ypred
    
        
    def evaluate(self, X, Y, use_sparse_solution=False):
        """Evaluates model in terms of validation criteria and standard error.
        
        Args:
            X: test data matrix, numpy array of shape (N, p).
            Y: test responses, numpy array of shape (N, ).
            use_sparse_solution: whether to use optimal solution or sparse solution, bool scalar.
        
        Returns:
            loss: float scalar.
            std_err: float scalar.
        """
        Ypred = self.predict(X, use_sparse_solution=use_sparse_solution).reshape(Y.shape)
        mse, rmse, mae, std_err = utils.metrics(Y, Ypred, y_preprocessor=self.y_scaler)
        return mse, rmse, mae, std_err
    
    def evaluate_and_save(self, Xtest, Ytest):
        """Evaluates optimal and sparse model in terms of validation criteria and standard error and logs results.
        
        Args:
            Xtest: test data matrix, numpy array of shape (Ntest, p).
            Ytest: test responses, numpy array of shape (Ntest, ).        
        """
        
        pen = np.array(['AMsWithInteractionsL0-opt','AMsWithInteractionsL0-sp'])
        M = pen.shape[0]
        df = pd.DataFrame(data={'': pen,
                                'Training {}'.format(self.eval_criteria): np.zeros(M), 
                                'Validation {}'.format(self.eval_criteria): np.zeros(M),
                                'Test {}'.format(self.eval_criteria): np.zeros(M),
                                'Test MSE': np.zeros(M),
                                'Test RMSE': np.zeros(M),
                                'Test MAE': np.zeros(M),
                                'Standard Error': np.zeros(M),
                                'Nonzeros': np.zeros(M)})
        df = df.set_index('')        

        train_mse_opt, train_rmse_opt, train_mae_opt, _ = self.evaluate(self.X, self.Y, use_sparse_solution=False)
        val_mse_opt, val_rmse_opt, val_mae_opt, _ = self.evaluate(self.Xval, self.Yval, use_sparse_solution=False)
        test_mse_opt, test_rmse_opt, test_mae_opt, std_err_opt = self.evaluate(Xtest, Ytest, use_sparse_solution=False)
        
        if self.eval_criteria == 'mse':
            train_eval_opt = train_mse_opt
            val_eval_opt = val_mse_opt
            test_eval_opt = test_mse_opt
        elif self.eval_criteria == 'mae':
            train_eval_opt = train_mae_opt
            val_eval_opt = val_mae_opt
            test_eval_opt = test_mae_opt
            
        with open(self.path+'/Results.txt', "a") as f:
            f.write('Optimal: Test-MSE: {:.6f}, Test-RMSE: {:.6f}, Test-MAE: {:.6f}, Standard-Error: {:.6f} \n'.format(test_mse_opt, test_rmse_opt, test_mae_opt, std_err_opt)) 
        print('Optimal: Test-MSE: {:.6f}, Test-RMSE: {:.6f}, Test-MAE: {:.6f}, Standard-Error: {:.6f}'.format(test_mse_opt, test_rmse_opt, test_mae_opt, std_err_opt))
    
        train_mse_sp, train_rmse_sp, train_mae_sp, _ = self.evaluate(self.X, self.Y, use_sparse_solution=True)
        val_mse_sp, val_rmse_sp, val_mae_sp, _ = self.evaluate(self.Xval, self.Yval, use_sparse_solution=True)
        test_mse_sp, test_rmse_sp, test_mae_sp, std_err_sp = self.evaluate(Xtest, Ytest, use_sparse_solution=True)
        if self.eval_criteria == 'mse':
            train_eval_sp = train_mse_sp
            val_eval_sp = val_mse_sp
            test_eval_sp = test_mse_sp
        elif self.eval_criteria == 'mae':
            train_eval_sp = train_mae_sp
            val_eval_sp = val_mae_sp
            test_eval_sp = test_mae_sp
        with open(self.path+'/Results.txt', "a") as f:
            f.write('Sparse: Test-MSE: {:.6f}, Test-RMSE: {:.6f}, Test-MAE: {:.6f}, Standard-Error: {:.6f}'.format(test_mse_sp, test_rmse_sp, test_mae_sp, std_err_sp)) 
        print('Sparse: Test-MSE: {:.6f}, Test-RMSE: {:.6f}, Test-MAE: {:.6f}, Standard-Error: {:.6f}'.format(test_mse_sp, test_rmse_sp, test_mae_sp, std_err_sp))    
        
        hp_opt = {'lam_sm': self.lam_sm_opt, 'lam_L0': self.lam_L0_opt}
        df.loc['AMsWithInteractionsL0-opt', 'Training {}'.format(self.eval_criteria)] = train_eval_opt
        df.loc['AMsWithInteractionsL0-opt', 'Validation {}'.format(self.eval_criteria)] = val_eval_opt
        df.loc['AMsWithInteractionsL0-opt', 'Test {}'.format(self.eval_criteria)] = test_eval_opt
        df.loc['AMsWithInteractionsL0-opt', 'Test MSE'], df.loc['AMsWithInteractionsL0-opt', 'Test RMSE'], df.loc['AMsWithInteractionsL0-opt', 'Test MAE'], df.loc['AMsWithInteractionsL0-opt','Standard Error'] = (test_mse_opt, test_rmse_opt, test_mae_opt, std_err_opt)
        df.loc['AMsWithInteractionsL0-opt', 'Nonzeros']=len(np.union1d(self.active_set_opt, np.unique(np.array(self.interaction_terms)[self.active_interaction_set_opt])))
        df.loc['AMsWithInteractionsL0-opt', 'Optimal Hyperparameters'] = ', '.join([f'{key}: {value}' for key, value in hp_opt.items()])
        hp_sp = {'lam_sm': self.lam_sm_sp, 'lam_L0': self.lam_L0_sp}
        df.loc['AMsWithInteractionsL0-sp', 'Training {}'.format(self.eval_criteria)] = train_eval_sp
        df.loc['AMsWithInteractionsL0-sp', 'Validation {}'.format(self.eval_criteria)] = val_eval_sp
        df.loc['AMsWithInteractionsL0-sp', 'Test {}'.format(self.eval_criteria)] = test_eval_sp
        df.loc['AMsWithInteractionsL0-sp', 'Test MSE'], df.loc['AMsWithInteractionsL0-sp', 'Test RMSE'], df.loc['AMsWithInteractionsL0-sp', 'Test MAE'], df.loc['AMsWithInteractionsL0-sp','Standard Error'] = (test_mse_sp, test_rmse_sp, test_mae_sp, std_err_sp)
        df.loc['AMsWithInteractionsL0-sp', 'Nonzeros']=len(np.union1d(self.active_set_sp, np.unique(np.array(self.interaction_terms)[self.active_interaction_set_sp])))
        df.loc['AMsWithInteractionsL0-sp', 'Optimal Hyperparameters'] = ', '.join([f'{key}: {value}' for key, value in hp_sp.items()])
        display(df)
        with open(os.path.join(self.path, 'AMsWithInteractionsL0.csv'), 'a') as f:
            df.to_csv(f, header=True, sep='\t', encoding='utf-8', index=True)
            
    def visualize_partial_dependences(self, X, Y, use_sparse_solution=False, saveflag=False):
        """Plot partial dependencies of main and interaction effects.
        
        TODO(shibal): Add interaction effects plots as well.
        
        Args:
            X: test data matrix, numpy array of shape (N, p).
            Y: test responses, numpy array of shape (N, ).        
        """
        if use_sparse_solution:
            beta = self.beta_sp
            delta = self.delta_sp
            active_set = self.active_set_sp
            active_interaction_set = self.active_interaction_set_sp
        else:
            beta = self.beta_opt
            delta = self.delta_opt
            active_set = self.active_set_opt
            active_interaction_set = self.active_interaction_set_opt

        # Generate b-splines on active set
        B, B_interaction = self.generate_splines_on_active_set(X, active_set, active_interaction_set)

        # Prediction
        Ypred = np.mean(self.Y) \
                + np.array(sum([B[j].dot(beta[j]) for j in active_set]))\
                + np.array(sum([B_interaction[j].dot(delta[j]) for j in active_interaction_set]))

        # Vizualize partial dependences
        if saveflag:
            folderpath = os.path.join(self.path, 'Figures')
            os.makedirs(folderpath, exist_ok=True)

        for k in notebook.tqdm(active_set, desc='Features'):
            plt.figure(figsize=(6, 6))
            plt.rc('font', **font)

            x_i_max = self.Xmax[k]
            x_i_min = self.Xmin[k]
            print('Feature:', repr(self.column_names[k]))
            ax1 = plt.gca()
            x_i = X[:, k]
            sort_indices = np.argsort(x_i)
            y_hat_i = B[k].dot(beta[k])
            Ypred -= y_hat_i        
            res = Y - Ypred
            ax1.scatter(x_i[sort_indices], res[sort_indices, 0], c='lightgrey', marker='.', label='$r$')

            y_hat_constant_i = B[k][:, 0:1].dot(beta[k][0:1,:])
            y_hat_nonlinear_i = B[k][:, 1:].dot(beta[k][1:,:])
            Ypred += y_hat_i

            ax1.plot(x_i[sort_indices], (y_hat_constant_i + y_hat_nonlinear_i)[sort_indices, 0], c='k', linewidth=2.0, label='$\\hat{r}$')
            ax1.legend()
            ax1.set_xlabel('$x_i$')
            ax1.set_ylabel('Partial Dependence')
            ax1.set_xlim(np.max([-2.5, np.min(x_i)]), 2.5)
            ax1.set_ylim(-25, 25)

            plt.tight_layout()
            if saveflag:
                plt.savefig(os.path.join(folderpath, '{}'.format(self.column_names[k])))
            plt.show() 

    def generate_x(self, i, j, percentile):
        """Generate data matrix with all covariates set to mean except for covariate i and j.
        
        Args:
            i: covariate i is varied over the full range, int scaler.
            j: covariate j is fixed to percentile value, int scaler.
            percentile: percentile used for covariate j, int scaler.
        """
        Xj_p = np.percentile(self.X[:, j], percentile)
        X_p = np.mean(self.X, axis=0, keepdims=True)
        X_p[:, j] = Xj_p
        n = 1000
        Xi = np.linspace(self.Xmin[i], self.Xmax[i], n)    
        X_p = X_p * np.ones((n, self.p))
        X_p[:, i] = Xi
        return X_p
    

    def interaction_effects_interpretability(self,
                                             percentile=25,
                                             use_sparse_solution=False,
                                             x_scaler=None):
        """Visualizing interactions for interpretability.
        
        Args:
            percentile: percentile to use for guaging effect of one of the covariates on the other in the interaction effect.
            use_sparse_solution: whether to use optimal solution or sparse solution, bool scalar.
            x_scaler: tuple of sklearn transformations, imputation followed by standardization for inverting the standardization
                for easier interpretability.
        """
        if use_sparse_solution:
            interaction_terms = self.interaction_terms[self.active_interaction_set_sp]
        else:
            interaction_terms = self.interaction_terms[self.active_interaction_set_opt]
        
        # Percentiles to consider
        lp = percentile
        hp = 100-lp

        for interaction_effect in notebook.tqdm(interaction_terms):

            i, j = interaction_effect
            print('i:', self.column_names[i],', j:', self.column_names[j])
            # Visualize effect of j on i
            # Create data matrix with xi, high/low xj and mean remaining covariates
            X_lp = self.generate_x(i, j, lp)
            X_hp = self.generate_x(i, j, hp)

            # Predict for low/high cases
            y_lp = self.predict(X_lp, use_sparse_solution=use_sparse_solution)   
            y_hp = self.predict(X_hp, use_sparse_solution=use_sparse_solution)   

            plt.figure(figsize=(7.5,7.5))
            font = {'weight' : 'bold',
                    'size'   : 12}
            plt.rc('font', **font)
            plt.plot(x_scaler[1].inverse_transform(X_lp)[:,i], y_lp, c='r', linewidth=2.0, label='Low - {}'.format(self.column_names[j]))
            plt.plot(x_scaler[1].inverse_transform(X_lp)[:,i], y_hp, c='C0', linewidth=2.0, label='High - {}'.format(self.column_names[j]))
            plt.ylabel('Self-Response', fontweight='bold', fontsize=12)
            plt.xlabel(self.column_names[i], fontweight='bold', fontsize=12)
            plt.legend()
            plt.show()

            # Visualize effect of i on j
            X_lp = self.generate_x(j, i, lp)
            X_hp = self.generate_x(j, i, hp)

            # Predict for low/high cases
            y_lp = self.predict(X_lp, use_sparse_solution=use_sparse_solution)   
            y_hp = self.predict(X_hp, use_sparse_solution=use_sparse_solution)   

            plt.figure(figsize=(7.5,7.5))
            plt.rc('font', **font)
            plt.plot(x_scaler[1].inverse_transform(X_lp)[:,j], y_lp, c='r', linewidth=2.0, label='Low - {}'.format(self.column_names[i]))
            plt.plot(x_scaler[1].inverse_transform(X_lp)[:,j], y_hp, c='C0', linewidth=2.0, label='High - {}'.format(self.column_names[i]))
            plt.ylabel('Self-Response', fontweight='bold', fontsize=12)
            plt.xlabel(self.column_names[j], fontweight='bold', fontsize=12)
            plt.legend()
            plt.show()        