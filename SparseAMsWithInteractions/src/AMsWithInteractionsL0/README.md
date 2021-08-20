## Additive Models with Interactions under L0

This folder contains the source files for running the additive model with interactions under L0

The following files serve the following purpose:
- AMsWithInteractionsL0.ipynb
- models.py (defines the AMI object which is called by the AMsWithInteractionsL0.ipynb notebook). 
- CrossValidation.py (Hyperparameter grid search with warm-starts over smoothness penalty: lambda_1). 
- L0path.py (Hyperparameter grid search with warm-starts over L0 regularization lambda_2 for each value of lambda_1).
- optimizer.py (Functions for running cyclic block coordinate descent over the main/interaction effects)
- utilities.py (B-Spline generation and quadratic penalties generation)