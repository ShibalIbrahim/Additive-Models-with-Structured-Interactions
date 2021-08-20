"""Nonparametric Additive Models with L0"""
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

from SparseAMsWithInteractions.src.AMsL0 import utilities
from SparseAMsWithInteractions.src.AMsL0 import CrossValidation
from SparseAMsWithInteractions.src.AMsL0 import L0Path
from SparseAMsWithInteractions.src.AMsL0 import CoordinateDescent
from SparseAMsWithInteractions.src import utils

os.environ['QT_QPA_PLATFORM']='offscreen'
font = {'weight' : 'bold',
        'size'   : 14}

class AM(object):
    """AMs with b-splines under L0 sparsity.
    
    Attributes:
        lams_sm: Regularization path over smoothness penalty for spline bases, float numpy 1D array. 
        lams_L0: Regularization path for L0 penalty for sparsity, float numpy 1D array.
        alpha: relative L0 penalty for interaction effects compared to main effects.
        max_support: Maximum support at which the regularization path is terminated, scalar int.
        convergence_tolerance: relative loss termination criteria for stopping, a float scalar.
        max_iter: maximum number of iterations for partially greedy Cyclic Coordinate Descent, int scalar
        eps: small epsilon added to QP for numerical stability for active set screening/updates, a float scalar.
        val_criteria: evaluation metric for hyperparameter tuning,
          - 'mae'
        val_crit_opt: optimal evaluation metric achieved.
        val_crit_sp: sparse evaluation metric achieved within 1% of optimal solution along L0 regularization path.
        
        X: training covariates, a float numpy array of shape (N, p).
        Y: training target responses, a float numpy array of shape (N,).

        p: number of covariates, int scalar.
    """
    def __init__(self,
                 lams_sm=np.logspace(start=-3, stop=-7, num=20, base=10.0),
                 lams_L0=np.logspace(start=0, stop=-4, num=25, base=10.0),
                 alpha=1.0,
                 max_support=1000,
                 convergence_tolerance=1e-4,
                 max_iter=1000,
                 eps=1e-8,
                 eval_criteria='mse',
                 degree=3,
                 active_set_update=False,
                 path=None,
                 logging=True,
                 terminate_val_L0path=True):
        assert path is not None
        os.makedirs(path, exist_ok=True)
        
        self.lams_sm = np.sort(lams_sm)[::-1]
        self.lams_L0 = np.sort(lams_L0)[::-1]
        self.alpha = alpha
        self.max_support = max_support
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
        self.active_set_update = active_set_update
    
    def load_data(self, X, Y, y_scaler, column_names, Xmin, Xmax, eps=1e-6):
        self.X = X
        self.Y = Y
        self.y_scaler = y_scaler
        self.column_names = column_names
        self.N, self.p = self.X.shape
        # eps added to give a slight margin at boundaries for spline generation
        self.Xmin = Xmin - eps 
        self.Xmax = Xmax + eps

    def generate_main_terms(self, generate_all=True, Mmax=10000, subsample=0.2, Ki=4):
        """Generates indices of main effects.
        
        Either generates all main effect indices or maximum screened top M effects
        Args:
            generate_all: whether to use generate all, boolean.
            Mmax: maximum number of main effects to be considered based on marginal fits, int scaler.
                ignored when generate_all=True.
            subsample: number of samples to consider for marginal fits, float scaler,
                ignored when generate_all=True.
            Kij: degrees of freedom for main terms, int scaler,
                ignored when generate_all=True.
        """
        self.generate_all = generate_all
        self.main_terms = np.arange(self.p)
        self.M = self.p
        self.Mmax = Mmax
        
        if not self.generate_all:
            assert Mmax <= self.M, "Mmax:{} must be less than maximum number of main effects:{}".format(Mmax, self.M)
            self.Mmax = Mmax
            num_cores = mp.cpu_count() 
            batches = num_cores
            batch_size = int(np.floor(self.M/batches))
            main_terms_batches = []
            for i in range(batches-1):
                main_terms_batches.append(self.main_terms[int(i*batch_size):int((i+1)*batch_size)])
            main_terms_batches.append(self.main_terms[int((batches-1)*batch_size):])

            idx = np.random.randint(self.N, size=int(np.ceil(subsample*self.N)))
            func = partial(utilities.screening, X=self.X[idx], Y=self.Y[idx], Xmin=self.Xmin, Xmax=self.Xmax, Ki=Ki) 

            with mp.Pool(num_cores) as pool:
                results = list(notebook.tqdm(pool.imap(func, main_terms_batches), total=len(main_terms_batches)))
                pool.close()
                pool.join()

            res_p = np.argsort([item for sublist in results for item in sublist])[:self.Mmax]
            self.main_terms = np.array([self.main_terms[k] for k in res_p])   
            
    def generate_splines_and_quadratic_penalties(self, Ki):
        """Generates b-splines and quadratic penalties and reduced BTB matrices.
        
        Ki: Degrees of freedom for b-spline basis, int scalar.
        """
        self.Ki = Ki
        self.Btrain, self.K_main = utilities.generate_bspline_transformed_X(self.X, self.Xmin, self.Xmax, self.Ki, self.main_terms)
        self.S = utilities.generate_bspline_quadratic_penalties(self.K_main)
        self.BtrainT_Btrain = [(B.transpose()).dot(B) for B in self.Btrain]

    def fitCV(self, Xval, Yval):
        """Runs Partially Greedy Cyclic Coordinate Descent with scalability considerations
        
        Fits models with Partially Greedy Cyclic Coordinate Descent with scalability considerations
        e.g. active set, cached matrix factorizations, warm-starts. The optimal solution is found over a grid search
        over a two-dimensional grid of lambda_1 and lambda_2.
        
        Args:
            Xval: validation covariates, a float numpy array of shape (Nval, p).
            Yval: validation target responses, a float numpy array of shape (Nval,).
        """
        self.Xval = Xval
        self.Yval = Yval
        CD_S_AS = (lambda Ypred, B, BT_B, P, S, I, beta, zeta, lam, active_set: CoordinateDescent.CD_Separate_ActiveSet(
            Ypred, B, BT_B, P, S, I, beta, zeta, lam, active_set, 
            Y=self.Y, main_terms=self.main_terms, max_iter=self.max_iter, tol=self.convergence_tolerance, path=self.path)
        )
        CD_S = (lambda CD_S_AS, Ypred, B, BT_B, P, S, I, beta, zeta, lam, active_set, full_set: CoordinateDescent.CD_Separate(
            CD_S_AS, Ypred, B, BT_B, P, S, I, beta, zeta, lam, active_set, full_set,
            Y=self.Y, main_terms=self.main_terms, max_iter=100, active_set_update=self.active_set_update, tol=self.convergence_tolerance, MaxSuppSize=self.max_support, path=self.path)
        )

        L0path = (lambda CD_S, CD_S_AS, lam_1, lams_2, active_set, beta, zeta :L0Path.L0Path(
            CD_S, CD_S_AS, lam_1, lams_2, active_set, beta, zeta,
            B=self.Btrain, BT_B=self.BtrainT_Btrain, K_main=self.K_main, Xval=self.Xval, Xmin=self.Xmin, Xmax=self.Xmax,
            Y=self.Y, Yval=self.Yval, y_scaler=self.y_scaler, S=self.S, main_terms=self.main_terms, eval_criteria=self.eval_criteria, path=self.path, logging=self.logging,  terminate_val_L0path=self.terminate_val_L0path))

        CV = (lambda L0path, CD_S, CD_S_AS: CrossValidation.CrossValidation(
            L0path, CD_S, CD_S_AS,
            B=self.Btrain, lams_1=self.lams_sm, lams_2=self.lams_L0, main_terms=self.main_terms, column_names=self.column_names, path=self.path, 
            logging=self.logging)
        )

        start = time.time()
        self.optimal_solution, self.sparse_solution, self.union_set, self.path_solution = CV(L0path=L0path, CD_S=CD_S, CD_S_AS=CD_S_AS)

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        if self.logging == True:
            with open(os.path.join(self.path, 'Results.txt'), "a") as f:
                f.write("Training completed in {:0>2}:{:0>2}:{:05.2f} \n".format(int(hours), int(minutes), seconds))
        print("Training completed in {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))   

        self.beta_opt, self.zeta_opt, self.lam_sm_opt, self.lam_L0_opt, self.J_opt, self.active_set_opt = self.optimal_solution
        self.beta_sp, self.zeta_sp, self.lam_sm_sp, self.lam_L0_sp, self.J_sp, self.active_set_sp = self.sparse_solution

    def generate_splines_on_active_set(self, X, active_set):
        """Generates splines on active set.
        
        Args:
            X: test data matrix, numpy array of shape (N, p).
            active_set: main effects to consider, int numpy array of shape (p, ).
        
        Returns:
            B: B-spline transformed matrices on active set, list of sparse matrices of shapes [(Ki+1,), ...].
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
        
        return B 
        
    def predict(self, X, use_sparse_solution=False): 
        """Generates spline transformations on new data and predicts the response.
        
        Args:
            X: test data matrix, numpy array of shape (N, p).
            use_sparse_solution: whether to use optimal solution or sparse solution, bool scalar.
        
        Returns:
            Ypred: numpy array of shape (N, ).
        """
        # Clip to handle covariate instances that maybe outside the spline basis generation
        for i in range(self.p):
            X[:,i] = np.clip(X[:,i], a_min=self.Xmin[i], a_max=self.Xmax[i]) 
        
        if use_sparse_solution:
            beta = self.beta_sp
            active_set = self.active_set_sp
        else:
            beta = self.beta_opt
            active_set = self.active_set_opt

        # Generate b-splines on active set
        B = self.generate_splines_on_active_set(X, active_set)

        # Prediction
        Ypred = np.mean(self.Y) + np.array(sum([B[j].dot(beta[j]) for j in active_set]))
        
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
        
        pen = np.array(['AMsL0-opt','AMsL0-sp'])
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
        
#         ntrials_bootstrap = 1000
#         test_mses_opt = np.zeros(ntrials_bootstrap)
#         test_rmses_opt = np.zeros(ntrials_bootstrap)
#         test_maes_opt = np.zeros(ntrials_bootstrap)
#         for i in range(ntrials_bootstrap):
#             idx = np.random.randint(Ytest.shape[0], size=Ytest.shape[0])
#             train_mses_opt[i], train_rmses_opt[i], train_maes_opt[i], _ = self.evaluate(self.X[idx], self.Y[idx], use_sparse_solution=False)
#             y_test_i = Ytest[idx, 0]
#             y_test_pred_i = Ytest_pred[idx, 0]
            
#             test_mses.append(self.(y_scaler.inverse_transform(y_test_i), y_scaler.inverse_transform(y_test_pred_i)))
#         standard_err = np.std(test_mses)
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
        df.loc['AMsL0-opt', 'Training {}'.format(self.eval_criteria)] = train_eval_opt
        df.loc['AMsL0-opt', 'Validation {}'.format(self.eval_criteria)] = val_eval_opt
        df.loc['AMsL0-opt', 'Test {}'.format(self.eval_criteria)] = test_eval_opt
        df.loc['AMsL0-opt', 'Test MSE'], df.loc['AMsL0-opt', 'Test RMSE'], df.loc['AMsL0-opt', 'Test MAE'], df.loc['AMsL0-opt','Standard Error'] = (test_mse_opt, test_rmse_opt, test_mae_opt, std_err_opt)
        df.loc['AMsL0-opt', 'Nonzeros']=len(self.active_set_opt)
        df.loc['AMsL0-opt', 'Optimal Hyperparameters'] = ', '.join([f'{key}: {value}' for key, value in hp_opt.items()])
        hp_sp = {'lam_sm': self.lam_sm_sp, 'lam_L0': self.lam_L0_sp}
        df.loc['AMsL0-sp', 'Training {}'.format(self.eval_criteria)] = train_eval_sp
        df.loc['AMsL0-sp', 'Validation {}'.format(self.eval_criteria)] = val_eval_sp
        df.loc['AMsL0-sp', 'Test {}'.format(self.eval_criteria)] = test_eval_sp
        df.loc['AMsL0-sp', 'Test MSE'], df.loc['AMsL0-sp', 'Test RMSE'], df.loc['AMsL0-sp', 'Test MAE'], df.loc['AMsL0-sp','Standard Error'] = (test_mse_sp, test_rmse_sp, test_mae_sp, std_err_sp)
        df.loc['AMsL0-sp', 'Nonzeros']=len(self.active_set_sp)
        df.loc['AMsL0-sp', 'Optimal Hyperparameters'] = ', '.join([f'{key}: {value}' for key, value in hp_sp.items()])
        display(df)

        with open(os.path.join(self.path, 'AMsL0.csv'), 'a') as f:
            df.to_csv(f, header=True, sep='\t', encoding='utf-8', index=True)
        
    def visualize_partial_dependences(self, X, Y, use_sparse_solution=False, saveflag=False):
        """Plot partial dependencies of main effects.
        
        Args:
            X: test data matrix, numpy array of shape (N, p).
            Y: test responses, numpy array of shape (N, ).        
        """
        if use_sparse_solution:
            beta = self.beta_sp
            active_set = self.active_set_sp
        else:
            beta = self.beta_opt
            active_set = self.active_set_opt

        # Generate b-splines on active set
        B = self.generate_splines_on_active_set(X, active_set)

        # Prediction
        Ypred = np.mean(self.Y) + np.array(sum([B[j].dot(beta[j]) for j in active_set]))

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
                plt.savefig(os.path.join(folderpath, '{}.pdf'.format(self.column_names[k])), bbox_inches='tight')
            plt.show() 
