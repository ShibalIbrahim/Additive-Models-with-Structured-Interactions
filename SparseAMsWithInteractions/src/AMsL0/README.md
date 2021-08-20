## Additive Models under L0

This folder contains the source files for running the additive model under L0 (Only main effects)

The following files serve the following purpose:
- AMsL0.ipynb (runs the model on 2019 Census Data)
- models.py (defines the AM object which is called by the AMsL0.ipynb notebook). 
- CrossValidation.py (Hyperparameter grid search with warm-starts over smoothness penalty: lambda_1). 
- L0path.py (Hyperparameter grid search with warm-starts over L0 regularization lambda_2 for each value of lambda_1).
- CoordinateDescent.py (Functions for running cyclic block coordinate descent over the covariate set)
- utilities.py (B-Spline generation and quadratic penalties generation)
