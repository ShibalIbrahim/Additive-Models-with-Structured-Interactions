## Additive Models with Interactions under L0

This folder contains the source files for running the additive model with interactions under L0

The following files serve the following purpose:
- GAMsWithInteractionsL0.ipynb (runs the model on 2019 Census Data)
- models.py (defines the GAMI object which is called by the GAMsWithInteractionsL0.ipynb notebook). 
- CrossValidation.py (Hyperparameter grid search with warm-starts over smoothness penalty: lambda_1). 
- L0path.py (Hyperparameter grid search with warm-starts over L0 regularization lambda_2 for each value of lambda_1).
- optimizer.py (Functions for running cyclic block coordinate descent over the main/interaction effects)
- utilities.py (B-Spline generation and quadratic penalties generation)

GAMsWithInteractionsL0-Subset (Runs on 40 features our additive model with interactions under L0) [to reproduce results for Figure 2b]
