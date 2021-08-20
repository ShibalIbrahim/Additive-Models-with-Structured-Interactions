## Additive Models with Interactions under Strong Hierarchy

This folder contains the source files for running the additive model with interactions under strong hierarchy

The following files serve the following purpose:
- GAMsWithInteractionsStrongHierarchy.ipynb (runs the model on 2019 Census Data)
- models.py (defines the GAMISH object which is called by the GAMsWithInteractionsStrongHierarchy.ipynb notebook). 
- CrossValidation_HS.py (Hyperparameter grid search with warm-starts over smoothness penalty: lambda_1). 
- L0path_HS.py (Hyperparameter grid search with warm-starts over L0 regularization lambda_2 for each value of lambda_1).
- HierarchicalSparsity.py (Uses Gurobi to solve the convex relaxation of the MIP under strong hierarchy constraints)
- ThresholdPath_HS.py (Generates sequence of feasible main/interaction effect subsets and solves the problem without any L0 constraints)
- utilities.py (B-Spline generation and quadratic penalties generation)
