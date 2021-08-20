# Baseline and Proposed Models

The folder contains source files for running baseline models and proposed models. The folders/files are listed below:

# Baseline Models:
The following files are used to run the baseline models
- Linear.ipynb (Ridge from sklearn, Lasso from sklearn, L0Learn from R using rpy2)
- LinearWithInteractions (Lasso from sklearn)
- NonInterpretable.py (xgboost with sklearn RandomizedSearchCV, feedforward NN using Tensorflow Keras with Hyperopt) 

# Proposed Models:
AMs under L0, AMs with Interactions under L0 and AMs with Interactions are strong hierachy are inside
GAMsL0, GAMsWithInteractionsL0 and GAMsWithInteractionsStrongHierarchy folders respectively

# Additional files:
- data_utils.py (contains preprocessing of Census data)
- utils.py (contains metrics: mse, rmse, mae)
- LinearWithInteractions-Subset (Runs on 40 features Lasso with all pairwise interactions using sklearn) [to reproduce Figure 2a]
- load_nn_model.py (creates a feedforward neural network model using Tensorflow Keras)

# Please verify the following tar files are unzipped (model.pkl files are generated, command: tar -xf model.pkl.tar.gz)
- GAMsL0/v1.0/model.pkl.tar.gz
- GAMsWithInteractionsL0/v1.0/r1.0/model.pkl.tar.gz [model.pkl needed by GAMsWithInteractionsL0.ipynb]
- GAMsWithInteractionsL0/v2.0/r1.0/model.pkl.tar.gz  [model.pkl needed by GAMsWithInteractionsStrongHierarchy.ipynb, Results.ipynb]
- GAMsWithInteractionsStrongHierarchy/v1.0/r1.0/model.pkl.tar.gz
